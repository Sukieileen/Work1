# 面向 Work1 当前 `BiMamba -> LinearAttention -> represents` 接口的 Latent MoE 设计方案

## 1. 设计目标

当前仓库中，`AttBiMambaModel` 先经过共享的 `BiMamba` 主干和 `LinearAttention`，再将 attention 权重作用到时序隐状态上做加权求和，得到序列级表示 `represents`，最后接一个 `NonLinear(sent_dim, 2)` 分类头。也就是说，`LinearAttention` 之后已经存在一个干净的 `[B, D]` 序列向量接口，适合替换成一个 sequence-level 的 MoE 分类层。  
参考当前代码：`hiddens -> sent_probs -> represents = (hiddens * sent_probs).sum(dim=1) -> self.proj(represents)`。  

本方案的原则是：

1. **不把 expert 绑定到具体系统**，而是采用 `N` 个 latent experts。
2. **路由粒度为 sequence-level**，不是 token-level。
3. **共享 backbone，MoE 放在 `represents` 之后**。
4. **用“强约束负载均衡 + 弱约束功能差异”**，避免 expert collapse，又不强行让 expert 在最终分类上互相唱反调。
5. **differentiable、好实现、和当前代码改动最小**。

---

## 2. 建议的 MoE 层结构

给定 attention 后的序列表示：

- `z ∈ R^{B×D}`
- `B` 为 batch size
- `D` 为 `sent_dim`

### 2.1 Router / Gate

先做一层轻量归一化，再计算 gate logits：

```text
z_norm = LN(z)
a = W_g z_norm + b_g          # a ∈ R^{B×N}
p = softmax(a / tau, dim=-1)  # p ∈ R^{B×N}
```

其中：

- `N` 是专家个数
- `tau` 是温度参数
- 推荐训练初期 `tau=1.5`，后期降到 `0.7~1.0`

### 2.2 Top-k 软路由

推荐 **top-k=2**。

对每个样本 `b`：

- 找到 `p_b` 中最大的两个 expert
- 构造 mask `m ∈ {0,1}^{B×N}`
- 归一化得到稀疏门控权重：

```text
p_tilde = (p ⊙ m) / (sum(p ⊙ m, dim=-1, keepdim=True) + eps)
```

这里：

- `⊙` 表示逐元素乘法
- `p_tilde ∈ R^{B×N}`
- 每个样本只有 2 个 expert 有非零权重

> 之所以不用 full-soft routing，是因为 full-soft 容易让所有 expert 都沾一点梯度但没人真正分工；  
> 之所以不用 top-1，是因为太容易早期塌缩，top-2 更稳。

### 2.3 Expert 结构

每个 expert 不建议一开始就做大 MLP，建议做 **bottleneck residual adapter + 轻量 head**。

对 expert `e`：

```text
h_e = W_down[e](z_norm)            # R^{B×r}
h_e = GELU(h_e)
delta_e = W_up[e](h_e)             # R^{B×D}
z_e = z + delta_e                  # residual expert representation
logit_e = Head_e(z_e)              # R^{B×C}
```

其中：

- `r` 是 bottleneck 维度
- `C` 是类别数，当前任务一般为 2
- `Head_e` 可以是 `Linear(D, C)` 或小两层 MLP

### 2.4 融合输出

最终 logit 取专家 logit 的加权和：

```text
logit = sum_e p_tilde[:, e] * logit_e
```

如果二分类采用 2-logit 交叉熵，则直接输出 `logit ∈ R^{B×2}`；如果采用 BCE，也可让 `Head_e` 输出 1 维。

> 推荐“融合 logit”，而不是“融合 feature 再接共享 head”。  
> 因为不同 expert 的价值更可能体现在**决策边界差异**，而不是仅仅 feature 空间小扰动。

---

## 3. 主损失

主损失保持现有分类目标即可：

- 若输出 2 维：`L_cls = CrossEntropy(logit, y)`
- 若输出 1 维：`L_cls = BCEWithLogits(logit, y)`

如果异常类别稀少，建议直接改成：

```text
L_cls = FocalBCE 或 class-balanced CE
```

---

## 4. 负载均衡损失：我的推荐方案

### 4.1 为什么不用只看一种均衡

只看 **importance**（平均概率质量）不够，因为一个 expert 可能只拿到少数几个样本，但 gate 给得很大。  
只看 **load**（被选中次数）也不够，因为一个 expert 可能被经常选中，但权重总是很小。

所以这里我建议同时约束：

- **importance**：平均概率质量是否均衡
- **load**：实际 top-k 选择频率是否均衡

这与早期稀疏 MoE 中对 `importance` 和 `load` 分别建模的思路一致；而 Switch Transformer 的简化辅助损失更适合 top-1 token routing。对我们这种 **sequence-level + top-2** 的场景，分开控制更稳。  

### 4.2 定义

设：

- `p ∈ R^{B×N}`：softmax 后的完整路由概率（**top-k 前**）
- `m ∈ {0,1}^{B×N}`：top-k 选择 mask，每个样本恰有 `k` 个 1
- `k=2`

#### (1) Importance 分布

```text
I_e = (1/B) * sum_b p_{b,e}
```

于是：

- `I ∈ R^N`
- `sum_e I_e = 1`

#### (2) Load 分布

把被选中次数归一化成概率分布：

```text
L_e = (1/(B*k)) * sum_b m_{b,e}
```

于是：

- `L ∈ R^N`
- `sum_e L_e = 1`

### 4.3 负载均衡损失公式

我推荐下面这个形式：

```text
L_imp = N * sum_e I_e^2 - 1
L_load = N * sum_e L_e^2 - 1
L_balance = 0.5 * (L_imp + L_load)
```

#### 解释

- 当 `I` 完全均匀时，`I_e = 1/N`，则 `L_imp = 0`
- 当 `L` 完全均匀时，`L_e = 1/N`，则 `L_load = 0`
- 越偏向少数 expert，这两个量越大

这个形式本质上就是“和均匀分布的二阶偏离度”，和用 `CV^2`（coefficient of variation squared）表达是等价思路，但实现更直接、数值更稳定。

### 4.4 为什么我选它，而不是别的

#### 不选纯 entropy 最大化

如果直接最大化 gate 熵，容易让每个样本都均匀分配到所有 expert：

- 看起来很“均衡”
- 实际上没有分工

这和我们想要的“top-k 稀疏分工”是冲突的。

#### 不只选 Switch 那个简化 dot-product loss

Switch 的负载均衡辅助损失是为 **top-1 token routing** 简化出来的。对于你这里的：

- sequence-level routing
- top-2 sparse routing
- 小规模 expert 数量

更自然也更稳定的是：

- importance 一项
- load 一项
- 分开约束

这样你能更清楚地看到到底是“概率质量塌缩”还是“实际分派塌缩”。

### 4.5 推荐系数

```text
lambda_balance = 1e-2 起步
可调范围：5e-3 ~ 5e-2
```

如果训练前期明显只用 1~2 个 expert，可逐渐提高；若发现 gate 虽均衡但分类性能下降，则降低。

---

## 5. Diversity Loss：我的推荐方案

## 核心观点

**不要对最终 logits 强行做“不同”约束。**

原因很简单：

- 对于很多 easy sample，所有 expert 本来就应该同意它是 normal 或 anomaly
- 如果你逼 logits 不一样，反而会伤害分类一致性和校准

所以 diversity 应该约束的是：

## **expert 的“变换方式 / residual function”不同**
而不是最终分类意见必须不同。

---

## 5.1 我推荐的 diversity loss：共激活条件下的残差去相关

对每个 expert，我们已经有：

```text
delta_e = W_up[e](GELU(W_down[e](z_norm)))   # R^{B×D}
```

也就是 expert 对共享表示 `z` 施加的残差修正。

我们先对每个样本的残差向量做 L2 归一化：

```text
delta_hat_{b,e} = delta_{b,e} / (||delta_{b,e}||_2 + eps)
```

然后，对每个样本 `b` 和专家对 `(e,f)`，定义余弦相似度：

```text
s_{b,e,f} = <delta_hat_{b,e}, delta_hat_{b,f}>
```

再定义“共激活权重”：

```text
c_{b,e,f} = p_tilde_{b,e} * p_tilde_{b,f}
```

最后 diversity loss 取：

```text
L_div = (1/B) * sum_b [ sum_{e<f} c_{b,e,f} * s_{b,e,f}^2 ]
```

### 解释

这项损失在做的事是：

- 如果两个 expert 在某个样本上**同时被 gate 赋予较高权重**
- 那就鼓励它们对该样本施加的 residual update 尽量**不共线 / 不冗余**

`s^2` 的好处是：

- `s=1` 和 `s=-1` 都算高度相关，都要罚
- 最优是在 `s≈0`，即近似正交

### 为什么这是我认为最合适的 diversity

#### 1. 约束的是“功能差异”，不是“结论差异”
我们让 expert 的**变换方向**不同，但不强迫它们给出相互矛盾的分类结果。

#### 2. 只在“共激活”时施压
如果两个 expert 根本不会同时被同一样本用到，就没必要硬逼它们不同。  
用 `c_{b,e,f}` 加权以后，差异性约束就只作用在**真正会竞争 / 共用责任**的 expert 对上。

#### 3. 不会过度惩罚 easy case 的一致判断
最终都判成 normal 没关系，只要它们内部处理路径不同即可。

---

## 5.2 为什么不推荐下面几种 diversity

### 不推荐 A：直接让 expert logits 彼此远离

问题：

- easy sample 上本来就该同意
- 会伤害分类稳定性
- 会伤害概率校准

### 不推荐 B：直接约束 expert 参数正交

问题：

- 参数不同不代表功能不同
- 参数相近也不代表功能相同
- 对 bottleneck MLP 这种小模块，参数级正交 often 太硬

### 不推荐 C：直接最大化 gate entropy 作为“diversity”

问题：

- 这是路由均匀，不是 expert 功能差异
- 甚至可能让所有样本都被平均分给所有 expert

---

## 5.3 推荐系数

```text
lambda_div = 1e-3 起步
可调范围：5e-4 ~ 1e-2
```

建议明显小于 `lambda_balance`。  
因为 balance 是“避免废 expert”的硬约束，diversity 是“避免重复 expert”的软约束。

---

## 6. 总损失

最终建议总损失：

```text
L_total = L_cls + lambda_balance * L_balance + lambda_div * L_div
```

如果后面发现 router logits 经常爆大，softmax 过早 one-hot，可再加一个很弱的 router z-loss：

```text
L_z = mean_b (logsumexp(a_b))^2
```

此时：

```text
L_total = L_cls + lambda_balance * L_balance + lambda_div * L_div + lambda_z * L_z
```

其中：

```text
lambda_z = 1e-4 或更小
```

但第一版可以先不加。

---

## 7. 推荐的训练策略

### Stage 0：warmup 单头模型

先训练当前单头：

```text
BiMamba -> LinearAttention -> NonLinear
```

得到一个稳定 backbone。

### Stage 1：替换成 MoE，冻结或半冻结 backbone

打开：

- gate
n- experts

但 backbone 先冻结 1~3 epoch，主要让：

- gate 学会基本路由
- expert 学会初步分工

### Stage 2：联合微调

- backbone 用更小 lr
- gate / expert 用更大 lr

例如：

```text
lr_backbone = 1e-4
lr_gate = 5e-4
lr_expert = 5e-4
```

这样更稳，不容易一开始让共享特征空间被 gate/expert 扰乱。

---

## 8. 推荐默认超参

### 结构

- `num_experts = 4` 或 `6`
- `top_k = 2`
- `bottleneck_dim = D/4`，若 D 较小可固定为 `32` 或 `64`
- `expert head = Linear(D, 2)` 或小 MLP

### 损失

- `lambda_balance = 1e-2`
- `lambda_div = 1e-3`
- `lambda_z = 0`（先不开）

### gate

- `tau = 1.5 -> 1.0` 逐步退火
- 可加轻微 dropout（如 0.1）

---

## 9. 伪代码

```python
# z: [B, D]
z_norm = layer_norm(z)

# router
router_logits = gate(z_norm)                # [B, N]
p = torch.softmax(router_logits / tau, dim=-1)

# top-k routing
vals, idx = torch.topk(p, k=top_k, dim=-1)
mask = torch.zeros_like(p).scatter(1, idx, 1.0)
p_tilde = p * mask
p_tilde = p_tilde / (p_tilde.sum(dim=-1, keepdim=True) + 1e-9)

# experts
expert_logits = []
expert_deltas = []
for e in range(N):
    h = gelu(W_down[e](z_norm))             # [B, r]
    delta = W_up[e](h)                      # [B, D]
    z_e = z + delta
    logit_e = head[e](z_e)                  # [B, C]
    expert_logits.append(logit_e)
    expert_deltas.append(delta)

expert_logits = torch.stack(expert_logits, dim=1)   # [B, N, C]
final_logit = (p_tilde.unsqueeze(-1) * expert_logits).sum(dim=1)

# classification loss
L_cls = criterion(final_logit, y)

# balance loss
importance = p.mean(dim=0)                              # [N], sum=1
load = mask.sum(dim=0) / (mask.sum() + 1e-9)           # [N], sum=1
L_imp = N * (importance ** 2).sum() - 1
L_load = N * (load ** 2).sum() - 1
L_balance = 0.5 * (L_imp + L_load)

# diversity loss
expert_deltas = torch.stack(expert_deltas, dim=1)      # [B, N, D]
expert_deltas = expert_deltas / (expert_deltas.norm(dim=-1, keepdim=True) + 1e-9)
L_div = 0.0
count = 0
for e in range(N):
    for f in range(e + 1, N):
        sim = (expert_deltas[:, e, :] * expert_deltas[:, f, :]).sum(dim=-1)   # [B]
        coact = p_tilde[:, e] * p_tilde[:, f]                                  # [B]
        L_div = L_div + (coact * sim.pow(2)).mean()
        count += 1
L_div = L_div / max(count, 1)

loss = L_cls + lambda_balance * L_balance + lambda_div * L_div
```

---

## 10. 最终结论

如果只让我拍板两件事：

### Load balance loss
我推荐：

```text
L_balance = 0.5 * [N * sum(I^2) - 1 + N * sum(L^2) - 1]
```

其中：

- `I` 用 top-k 前 soft 概率计算
- `L` 用 top-k 后实际 mask 计算

这是当前这个 **sequence-level latent MoE** 最稳妥的平衡写法。

### Diversity loss
我推荐：

```text
L_div = mean_b sum_{e<f} [ p_tilde_{b,e} p_tilde_{b,f} * cos(delta_{b,e}, delta_{b,f})^2 ]
```

即：

- 不逼最终分类结论不同
- 只逼共激活 expert 的 residual 变换别学成一个东西

这是一个“够强但不乱来”的差异性约束。

---

## 11. 相关依据

- 当前仓库 `AttBiMambaModel` 在 `LinearAttention` 后将注意力权重作用于 `hiddens` 并求和得到 `represents`，再接 `NonLinear(sent_dim, 2)` 分类头，因此在 `represents` 位置替换为 MoE 层改动最小。  
- `LinearAttention` 的输出是每个序列位置的 attention 权重，因此 sequence-level MoE 最自然的接口是 attention 加权后的 `represents`。  
- 稀疏 MoE 经典工作会分别考虑 `importance` 与 `load` 两类平衡；而 Switch Transformer 的简化辅助损失是围绕 top-1 routing 给出的。对这里的 top-2 sequence routing，分开约束更合适。  

