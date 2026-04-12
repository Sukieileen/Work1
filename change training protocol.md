# 从 MetaLog 到 BiMamba+Attention+Latent-MoE 的训练协议（对齐原文数据设定）

## 1. 目标

本文档给出一个**严格对齐 MetaLog 原文数据使用方式**、但**不再使用整网元学习**的训练协议，用于当前的：

- `BiMamba -> LinearAttention -> represents -> Latent MoE -> classifier`

核心思想是：

1. **保持 MetaLog 原文的 source/target 数据规模设定不变**；
2. **放弃 bilevel meta-learning**，改为普通监督训练；
3. **保留 source 多、target 少的不对称结构**；
4. **用联合训练中的 target loss 权重来替代 MetaLog 中 meta-test loss 的作用**；
5. **让 target 每一步都参与更新，但不追求 source/target 样本量平衡**。

---

## 2. 与 MetaLog 原文保持一致的数据设定

这里以 `MetaLog` 仓库代码实现为准，而不是按口头近似描述理解。原版的关键不是“把 normal/anomaly 两类样本分别独立按比例切出来”，而是：

1. 先按原始实例顺序切出一个 `train+dev` 前缀；
2. 只在这个前缀内部打乱，得到 `train/dev`；
3. 对 target 域，只在 `train` 里再做一次异常保留；
4. `dev` 保留出来，但原始训练脚本基本没有真正消费它。

### 2.1 HDFS -> BGL

按原版 `cut_by_415` 与 `cut_by_316_filter`：

- Source(HDFS): `40% train / 10% dev / 50% unused`
- Target(BGL): `30% train / 10% dev / 60% test`
- Target 异常暴露规则：只在 target `train` 内保留 `1%` 异常，因此对整份 BGL 来说，模型实际看到的是“大约 `30%` normal + `0.3%` anomaly”，不是“整库 `1%` anomaly”

当前协议如果想和原版公平比较，target 只能暴露到这个量级。

### 2.2 BGL -> HDFS

按原版 `cut_all` 与 `cut_by_253_filter`：

- Source: `all BGL sessions`
- Target(HDFS): `20% train / 50% dev / 30% test`
- Target 异常暴露规则：只在 target `train` 内保留 `1%` 异常，因此对整份 HDFS 来说，模型实际看到的是“大约 `20%` normal + `0.2%` anomaly”

这也是之前协议里最不公平的地方之一：如果直接按“`10% normal + 1% anomaly` 整库独立采样”，会同时少看 normal、又多看 anomaly。

### 2.3 原则

后续所有训练协议都必须满足：

- **不额外增加 target 训练数据量**；
- **不把 target 的 anomaly 暴露量从“train 内 1% 保留”偷换成“整库 1% 抽样”**；
- **不把 target 全量无标注数据并入训练**；
- **不做人为的 source/target 样本数强行平衡**；
- **仅在训练策略上改造，不在数据使用量上破坏与 MetaLog 的可比性**；
- **如果要利用保留下来的 dev split，也要单独论证，因为那会让模型额外获得 target 域监督信息**。

---

## 3. 新模型结构

当前模型接口约定为：

```text
Embedding
 -> BiMamba
 -> LinearAttention
 -> represents            # [B, D]
 -> Latent MoE
 -> final logit
```

其中：

- `represents` 是 attention 加权汇聚后的序列级表示；
- MoE 放在 `represents` 之后；
- expert 不与 system id 显式绑定；
- gate 基于 `represents` 做 sequence-level routing。

---

## 4. 为什么不再使用整网元学习

原 MetaLog 的核心是：

- 单一共享网络；
- source split 做 meta-train；
- target split 做 meta-test；
- 通过 bilevel optimization 强化 source -> target 泛化。

但在当前结构中，参数已经被拆成：

- shared backbone（Embedding/BiMamba/Attention）
- gate
- latent experts

如果继续做整网元学习，会导致：

1. gate 与 experts 互相抢解释权；
2. 优化目标耦合过重；
3. 很难判断增益来自 MoE 还是来自元学习；
4. target 数据极少时，容易过拟合到 meta-loop 的细节。

因此采用**普通监督训练 + 分阶段训练**更合适。

---

## 5. 总体训练流程

推荐采用三阶段：

1. **Phase A: source-only warmup**
2. **Phase B: source/target joint fine-tune**
3. **Phase C: target calibration**

即：

```text
source 预热主干
    ->
引入少量 target 联合训练 MoE
    ->
最后用 target 小规模校准 gate / experts
```

---

## 6. Phase A：Source-only Warmup

### 6.1 目标

只使用 source 数据，先训练稳定的共享表示。

### 6.2 模型

推荐先不用 MoE，或者临时让 MoE 退化为单头：

```text
Embedding -> BiMamba -> LinearAttention -> single head
```

### 6.3 训练数据

- 只用 source 训练集；
- HDFS->BGL 时只用 `40% HDFS`；
- BGL->HDFS 时只用 `all BGL`。

### 6.4 损失

使用普通分类损失：

- 二分类交叉熵（BCE/CE）
- 或 focal loss（更推荐）

记为：

```text
L_warm = L_src
```

### 6.5 参数更新

更新：

- Embedding
- BiMamba
- LinearAttention
- single classifier head

### 6.6 建议

- 训练到验证集基本收敛即可；
- 不追求此阶段最终最优，只追求 backbone 表示稳定。

---

## 7. Phase B：Source/Target Joint Fine-tune（核心阶段）

这是最关键的训练阶段。

### 7.1 目标

在保持 source 主导学习的前提下，让 target 小样本在**每一步更新中都持续发挥作用**。

### 7.2 关键原则

这里**不做样本量平衡**，而做：

- **双 dataloader 联合训练**
- **loss 层面的 source/target 权重平衡**

也就是说：

- source 仍然多；
- target 仍然少；
- 但每个 step 同时看到 source 和 target。

### 7.3 数据加载方式

每一步同时采样：

- 一个 `source batch`
- 一个 `target batch`

#### 推荐 batch 构造

##### HDFS -> BGL

- `batch_source = 64`
- `batch_target = 16` 或 `32`

##### BGL -> HDFS

- `batch_source = 64`
- `batch_target = 16` 或 `32`

### 7.4 target batch 采样策略

由于 target 非常少，建议：

- **with replacement** 采样；
- 每轮都重复采样 target 训练样本；
- target 不要求 epoch 内遍历一次算一轮。

这样做的本质是：

- target 不靠数据量影响模型；
- target 靠“每一步都有梯度贡献”影响模型。

### 7.5 模型

切换到完整结构：

```text
Embedding
 -> BiMamba
 -> LinearAttention
 -> represents
 -> Gate(represents)
 -> Latent Experts(represents)
 -> fused logit
```

### 7.6 总损失

总损失定义为：

```text
L_total = L_src + λ_tgt * L_tgt + λ_bal * L_balance + λ_div * L_div
```

其中：

- `L_src`：source batch 上的分类损失
- `L_tgt`：target batch 上的分类损失
- `L_balance`：MoE load balance loss
- `L_div`：MoE diversity loss

### 7.7 target 权重 λ_tgt

推荐：

```text
λ_tgt = 4
```

原因：

- MetaLog 原文中 meta-test loss 的权重超参 `β=4` 最优；
- 当前不再使用元学习，但 `λ_tgt` 可以视为对原来 target 影响力的一种普通训练化改写；
- 因此建议先从 `4` 起步，再在 `{2, 4, 6}` 上做小范围调参。

### 7.8 参数更新策略

此阶段建议：

- shared backbone：小学习率
- gate：正常学习率
- experts：正常学习率

#### 建议学习率

举例：

- backbone lr = `1e-4`
- gate lr = `5e-4`
- experts lr = `5e-4`

如果 backbone 仍较不稳定，可进一步减到：

- backbone lr = `5e-5`

### 7.9 推荐冻结策略

不建议完全冻结 backbone，但建议“慢更新”：

- Embedding / BiMamba / LinearAttention：小 lr
- Gate / Experts：正常 lr

这样可以：

- 保留 Phase A 学到的通用表示；
- 让 MoE 分工成为主要适配来源。

---

## 8. Phase C：Target Calibration

### 8.1 目标

在不引入额外 target 数据的前提下，对目标域进行小规模校准。

### 8.2 训练数据

只使用与原文相同的那一小撮 target 训练数据：

- HDFS->BGL: `30% BGL train slice + train 内 1% anomaly 保留`
- BGL->HDFS: `20% HDFS train slice + train 内 1% anomaly 保留`

### 8.3 更新参数

只更新：

- gate
- expert heads / expert adapters
- 可选 temperature / threshold 参数

冻结：

- Embedding
- BiMamba
- LinearAttention

### 8.4 损失

```text
L_calib = L_tgt + λ_bal_small * L_balance
```

注意：

- 这阶段 `L_div` 可以关掉或减弱；
- 重点是微调决策边界与路由，而不是重新塑造专家分工。

### 8.5 建议

- epoch 数要少，例如 `3~5`；
- 使用 early stopping；
- 防止对极少量 target 过拟合。

---

## 9. 为什么不做“source/target 样本数平衡”

### 9.1 不符合 MetaLog 原文设定

原文本来就是：

- source 多
- target 少
- target 只是少量引导

不是对称训练。

### 9.2 容易导致 target 过拟合

如果把 target 采样到和 source 数量接近，实际上是在重复利用同一小批 target 样本，带来：

- 记忆 target 训练集；
- decision boundary 偏向小样本噪声；
- 泛化能力下降。

### 9.3 真正需要平衡的是“梯度存在感”

因此更合理的做法是：

- target 每步都参与；
- 但 target 影响力通过 `λ_tgt` 控制；
- 而不是通过样本数量堆出来。

---

## 10. 推荐 sampler 设计

### 10.1 source sampler

正常随机采样即可，但建议：

- source 内部做 anomaly/normal 类平衡；
- 或在 loss 里用 focal/class-balanced。

### 10.2 target sampler

必须支持：

- with replacement
- class-aware sampling

因为 target anomaly 极少，否则一个 epoch 很快把 anomaly 用光。

### 10.3 pairwise batch 结构

每步训练结构为：

```text
(source_batch, source_label), (target_batch, target_label)
```

然后：

```text
logit_src = model(source_batch)
logit_tgt = model(target_batch)

L_src = cls(logit_src, y_src)
L_tgt = cls(logit_tgt, y_tgt)
L_total = L_src + λ_tgt * L_tgt + λ_bal * L_balance + λ_div * L_div
```

---

## 11. 推荐的默认超参数

### 11.1 训练阶段

- Warmup epochs: `5 ~ 10`
- Joint fine-tune epochs: `10 ~ 20`
- Calibration epochs: `3 ~ 5`

### 11.2 损失权重

- `λ_tgt = 4`
- `λ_bal = 1e-2`
- `λ_div = 1e-3`

### 11.3 优化器

- AdamW

### 11.4 学习率

#### Phase A
- backbone/head: `1e-4 ~ 5e-4`

#### Phase B
- backbone: `5e-5 ~ 1e-4`
- gate: `5e-4`
- experts: `5e-4`

#### Phase C
- gate: `1e-4 ~ 5e-4`
- experts/head: `1e-4 ~ 5e-4`
- backbone: frozen

---

## 12. 最终推荐协议（简洁版）

### HDFS -> BGL

1. 用 `40% HDFS` 做 source-only warmup
2. 用：
   - source = `40% HDFS`
   - target = `30% BGL train slice + train 内 1% anomaly 保留`
   做 joint fine-tune
3. 每步：
   - 一个 source batch
   - 一个 target batch（有放回）
4. 损失：

```text
L = L_src + 4 * L_tgt + λ_bal * L_balance + λ_div * L_div
```

5. 最后只用 target 小集做 calibration

### BGL -> HDFS

1. 用 `all BGL` 做 source-only warmup
2. 用：
   - source = `all BGL`
   - target = `20% HDFS train slice + train 内 1% anomaly 保留`
   做 joint fine-tune
3. 每步：
   - 一个 source batch
   - 一个 target batch（有放回）
4. 损失同上
5. 最后做 target calibration

---

## 13. 与原 MetaLog 的关系

### 保留的部分

- source abundant, target scarce 的设定
- target 只用少量 normal + 极少 anomaly
- target 在训练中必须持续参与

### 改掉的部分

- 不再做 meta-train / meta-test 的 bilevel optimization
- 不再构造 meta-task
- 不再用单一共享网络承担全部跨域泛化

### 新方案的本质

- 用普通监督训练替代元学习
- 用 `λ_tgt * L_tgt` 替代原来 meta-test 分支的 target 牵引作用
- 用 MoE 的 gate/expert 分工承担跨域适配的主要责任

---

## 14. 一句话总结

当前最合理的方案不是：

- 先训 HDFS 再训 BGL；
- 也不是把 source/target 样本数强行平衡；
- 更不是继续原封不动使用整网元学习。

而是：

## **严格沿用 MetaLog 的 target 数据量设定，采用 source-only warmup + source/target joint fine-tune + target calibration 的普通监督训练协议。**
