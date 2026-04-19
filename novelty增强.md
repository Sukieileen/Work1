# Work1 下一步方法设计与代码改造（面向当前仓库的可落地版本）

## 1. 这次先做什么，不做什么

这次建议只做两件事：

1. **Normality Anchoring（正常模式锚定）**
2. **Drift-aware MoE（带漂移感知的专家路由）**

先**不做**大规模实验扩展，也**不做** conformal calibration。实验仍然只保留你现在最核心的两个方向：

- HDFS -> BGL
- BGL -> HDFS

也就是说，这一版目标不是把系统改得很花，而是把你现有的：

- parser-free 表示
- BiMamba / GRU 主干
- attention pooling
- latent MoE
- staged training

统一到一个更强的中心故事里：

> **在跨系统迁移下，模型应优先学到可迁移的 normality 边界，而不是过拟合少量已见异常。**

---

## 2. 总体方法：从“分类器”改成“normality-aware detector”

你现在的主线本质上还是：

- 编码得到表示 `h`
- 交给 `proj` / `MoE proj`
- 直接输出二分类结果

我建议改成下面这个结构：

```text
parser-free / template embedding
        -> sequence backbone (GRU or BiMamba)
        -> attention pooling
        -> pooled representation h
        -> drift-aware MoE
            -> expert-specific representation h_k
            -> expert-specific prototype distance d_k
            -> expert-specific classification logits z_k
        -> fused anomaly logits
```

关键变化是：

### 原来
你让模型只学“这个样本像不像 anomaly”。

### 现在
你让模型同时学两件事：

1. 这个样本是不是 anomaly；
2. 这个样本离“正常模式中心”有多远。

这样论文故事就从“架构组合”升级成“**normality transfer**”。

---

## 3. 方法公式设计

## 3.1 表示层

对于一个日志序列样本 `x_i`，经过现有 backbone 与 attention pooling 后得到序列级表示：

\[
h_i \in \mathbb{R}^{d}
\]

这里 `h_i` 对应你现在 `GRU/BiMamba + attention` 之后的 pooled representation，也就是当前 `self.proj(...)` 之前的那一层表示。

这一步不改主干，只改其后的判别机制。

---

## 3.2 全局 normal prototype 与 expert prototype

定义：

- 全局正常原型：
\[
p_g \in \mathbb{R}^{d}
\]

- 第 `k` 个 expert 的正常原型：
\[
p_k \in \mathbb{R}^{d}, \quad k = 1,2,\dots,K
\]

其中：

- `p_g` 负责建模“跨系统共享的正常模式中心”；
- `p_k` 负责建模“某一种正常模式子空间 / 某类漂移场景下的正常边界”。

---

## 3.3 漂移感知路由

你现有 router 只看输入表示 `h_i`。现在改成同时看：

- 表示本身；
- 与全局 prototype 的距离；
- 表示范数；
- 一个简单的不确定性 / 边界感知统计量。

先定义一个轻量 router feature：

\[
\phi_i = \big[h_i;\, \|h_i - p_g\|_2^2;\, \|h_i\|_2\big]
\]

然后 router logits 为：

\[
r_i = W_r \cdot \text{LN}(\phi_i) + b_r
\]

路由概率：

\[
q_i = \text{softmax}(r_i / \tau)
\]

再做当前已有的 top-k 稀疏选择。

### 为什么这样改

这样 router 不再只是“看 latent feature 分专家”，而是：

> **根据样本偏离 normality 的程度，把它分配到更合适的 expert。**

这会显著增强 MoE 的可解释性，也更贴合“cross-system drift”这个故事。

---

## 3.4 expert 表示与 prototype-aware logit fusion

你当前每个 expert 的表示可以继续保留：

\[
\tilde{h}_{i,k} = h_i + U_k\big(\text{GELU}(D_k h_i)\big)
\]

也就是继续沿用当前 down-proj / up-proj 的 delta 形式。

然后定义该 expert 下的 prototype distance：

\[
d_{i,k} = \|\tilde{h}_{i,k} - p_k\|_2^2
\]

原始 expert 分类头输出二分类 logits：

\[
z^{cls}_{i,k} = f_k(\tilde{h}_{i,k}) \in \mathbb{R}^{2}
\]

再构造一个 prototype-induced logits：

\[
z^{proto}_{i,k} = \alpha \cdot [-d_{i,k},\, d_{i,k}]
\]

其中：

- 距离越小，越偏向 normal；
- 距离越大，越偏向 anomaly。

于是 expert 最终 logits 为：

\[
z_{i,k} = z^{cls}_{i,k} + z^{proto}_{i,k}
\]

最后总 logits 为：

\[
z_i = \sum_{k=1}^{K} q_{i,k} z_{i,k}
\]

### 这一设计的优点

这个改法最大的好处是：

- **不需要推翻现有二分类头**；
- 只是给它加了一个“基于 normality 距离的偏置项”；
- 推理阶段照样输出二分类 logits，和你现在的评测流程兼容。

---

## 3.5 Prototype loss

为了让 prototype 真学到“正常模式”，建议增加两个 loss。

### 1）正常样本拉近损失
对 normal 样本：

\[
\mathcal{L}_{pull} = \frac{1}{|\mathcal{N}|}
\sum_{i \in \mathcal{N}}
\Big(
\|h_i - p_g\|_2^2
+ \sum_{k=1}^{K} q_{i,k} \|\tilde{h}_{i,k} - p_k\|_2^2
\Big)
\]

### 2）异常样本推远损失
对 anomaly 样本，做 margin-based push：

\[
\mathcal{L}_{push} = \frac{1}{|\mathcal{A}|}
\sum_{i \in \mathcal{A}}
\Big(
\max(0, m_g - \|h_i - p_g\|_2)^2
+ \sum_{k=1}^{K} q_{i,k} \max(0, m_k - \|\tilde{h}_{i,k} - p_k\|_2)^2
\Big)
\]

### 3）专家 prototype 分离正则
防止所有 expert prototype 塌缩到一起：

\[
\mathcal{L}_{sep} = \frac{1}{K(K-1)} \sum_{a \neq b}
\cos^2(p_a, p_b)
\]

---

## 3.6 总损失

### 单域 warmup 阶段

\[
\mathcal{L}_{warm} = \mathcal{L}_{cls} + \lambda_{aux}\mathcal{L}_{moe} + \lambda_{proto}(\mathcal{L}_{pull}+\mathcal{L}_{push}) + \lambda_{sep}\mathcal{L}_{sep}
\]

### source + target joint 阶段

\[
\mathcal{L}_{joint} = \mathcal{L}^{src}_{cls} + \beta\mathcal{L}^{tgt}_{cls}
+ \lambda_{aux}\mathcal{L}_{moe}
+ \lambda_{proto}^{src}\mathcal{L}^{src}_{proto}
+ \lambda_{proto}^{tgt}\mathcal{L}^{tgt}_{proto}
+ \lambda_{sep}\mathcal{L}_{sep}
\]

其中：

- `β` 继续沿用你现在的 `target_weight` 逻辑；
- `L_proto^tgt` 可以先只对 target 中的 normal 样本生效，这样更符合“normality transfer”的主线。

---

## 4. 和当前仓库怎么对接

## 4.1 当前仓库里，哪些地方是现成可复用的

你现在仓库已经有几块特别适合直接接这个设计：

1. `representations/parser_free.py` 里 parser-free 编码器是冻结的 `AutoModel`，并用 `mean/cls` pooling 输出静态事件表示；这意味着我们这次可以把创新集中在检测头，而不是再去碰 PLM 表示层。  
2. `models/gru.py` 和 `models/mamba.py` 都是先得到 sequence hidden states，再用 attention pooling 得到序列表示，最后交给 `self.proj`；这正好给 prototype-aware classifier 留出了稳定插口。  
3. `models/moe.py` 里的 `LatentMoEClassifier` 已经有 router、top-k、expert delta、auxiliary loss；我们只需要把它从“纯 latent 路由器”升级成“带 normality 感知的 MoE”。  
4. `approaches/supervised_protocol.py` 里当前损失仍是 `source_loss + target_weight * target_loss + aux_loss`，因此加入 prototype loss 的入口非常清楚。  

---

## 5. 代码改造点（按文件拆）

## 5.1 新增文件：`models/normality.py`

建议新建一个小文件，不要把所有 prototype 逻辑都塞进 `moe.py`。

### 建议新增类

```python
class NormalPrototypeBank(nn.Module):
    def __init__(self, input_dim, num_experts, margin_global=1.0, margin_expert=1.0):
        ...

    def global_distance(self, h):
        ...

    def expert_distance(self, expert_h):
        ...

    def compute_loss(self, h, expert_h, routing_probs, targets, anomaly_id):
        ...

    def get_metrics(self):
        ...
```

### 它负责什么

- 管理 `global prototype`
- 管理 `expert prototypes`
- 计算 `pull / push / sep loss`
- 输出 prototype 相关指标

### 为什么拆出去

这样 `moe.py` 只负责：

- 路由
- 专家变换
- logits 融合

而 `normality.py` 只负责：

- prototype 参数与 loss

这样结构更清晰，后面写论文也更好解释。

---

## 5.2 修改 `models/moe.py`

当前 `LatentMoEClassifier` 的核心结构已经很完整，不要推翻。建议做“增强”，不要重写。

### 当前已有结构

当前类里已经有：

- `self.router = nn.Linear(input_dim, num_experts)`
- `down_projs / up_projs`
- `heads`
- `balance/diversity/z-loss`

### 建议改法

#### 1）构造 drift-aware router 输入
把现在：

```python
router_logits = self.router(router_inputs)
```

改成：

```python
router_features = concat([normalized_inputs, global_distance, feature_norm])
router_logits = self.router(router_features)
```

因此 `self.router` 的输入维度要从：

- `input_dim`

改成：

- `input_dim + extra_router_dim`

#### 2）在每个 expert 后面计算 prototype distance
现在每个 expert 都会产生：

```python
expert_representation = inputs + delta
expert_logits.append(head(expert_representation))
```

建议扩成：

```python
cls_logits = head(expert_representation)
proto_logits = alpha * torch.stack([-dist, dist], dim=-1)
expert_logits.append(cls_logits + proto_logits)
```

#### 3）在 `forward` 中缓存中间量
为了让训练器在外面计算 loss，需要缓存：

- `normalized_inputs`
- `expert_representations`
- `routing_probs`
- `routing_mask`
- `global_distance`

例如：

```python
self._last_cache = {
    'base_repr': normalized_inputs,
    'expert_repr': expert_representations,
    'routing_probs': sparse_probs,
}
```

#### 4）增加接口
新增：

```python
def get_prototype_loss(self, targets, anomaly_id):
    ...

def get_prototype_metrics(self):
    ...
```

这样外部训练器不用侵入 forward 签名。

### 一个重要原则

**不要把 targets 直接塞进 forward。**

因为你现在整个仓库默认 `model(inputs) -> logits`，如果改 forward 签名，连 `predict`、`evaluate`、`collect_anomaly_scores` 都要一起动，破坏面太大。

---

## 5.3 修改 `models/gru.py`

你现在 GRU 模型的关键流程是：

```python
hiddens -> attention -> represents -> self.proj(represents)
```

### 建议改法

#### 1）增加一个中间函数
把 pooled representation 的生成单独抽出来：

```python
def encode_representation(self, inputs):
    ...
    return represents
```

然后 `forward` 里只做：

```python
represents = self.encode_representation(inputs)
outputs = self.proj(represents)
return outputs
```

这样做的好处是后面如果你想做表示分析、prototype 可视化、t-SNE，都方便。

#### 2）增加 pass-through 接口
加上：

```python
def get_prototype_loss(self, targets, anomaly_id):
    if hasattr(self.proj, 'get_prototype_loss'):
        return self.proj.get_prototype_loss(targets, anomaly_id)
    return self.atten_guide.new_zeros(())


def get_prototype_metrics(self):
    if hasattr(self.proj, 'get_prototype_metrics'):
        return self.proj.get_prototype_metrics()
    return {}
```

### 不需要改的部分

- GRU 主干本身不需要改；
- attention pooling 不需要改；
- embedding 也不需要动。

---

## 5.4 修改 `models/mamba.py`

这里和 `gru.py` 思路完全一样。

### 建议改法

#### 1）抽 `encode_representation`
把：

- embedding
- input projection
- BiMamba layers
- attention pooling

这一串封装成：

```python
def encode_representation(self, inputs):
    ...
    return represents
```

#### 2）新增 prototype pass-through 接口
同 `gru.py`。

### 为什么 GRU / BiMamba 两边都要这样改

因为你现在论文故事是统一的，不能让新方法只适配某一个 backbone。哪怕最后主结果只打 BiMamba，也最好在代码结构上让 GRU 能跑同一套逻辑，这样整个仓库更规整。

---

## 5.5 修改 `approaches/supervised_protocol.py`

这个文件是这次最关键的训练入口。

### 当前逻辑

现在有两条最核心的损失路径：

#### 单 batch

```python
total_loss = cls_loss + aux_loss
```

#### joint batch

```python
total_loss = source_loss + target_weight * target_loss + aux_loss
```

### 建议改法

#### 1）在 `MetaLog.__init__` 里新增参数
建议新增：

```python
use_normality_anchor=False,
prototype_scale=1.0,
prototype_loss_weight=0.1,
prototype_sep_weight=1e-3,
prototype_margin_global=1.0,
prototype_margin_expert=1.0,
prototype_target_normal_only=True,
router_use_distance=True,
```

#### 2）增加 prototype loss 收集函数
新增两个小函数：

```python
def _prototype_loss(self, targets):
    if hasattr(self.model, 'get_prototype_loss'):
        return self.model.get_prototype_loss(targets, self.label2id['Anomalous'])
    return targets.new_zeros((), dtype=torch.float32)


def _prototype_metrics(self):
    if hasattr(self.model, 'get_prototype_metrics'):
        return self._scalarize_metrics(self.model.get_prototype_metrics())
    return {}
```

#### 3）修改 `compute_single_batch_loss`
改成：

```python
proto_loss = self._prototype_loss(tinst.targets)
total_loss = cls_loss + aux_loss + self.prototype_loss_weight * proto_loss
```

#### 4）修改 `compute_joint_batch_loss`
当前 source / target 是分开的，所以这里要分别算 prototype loss：

```python
source_proto_loss = ...
target_proto_loss = ...
```

建议总式写成：

```python
total_loss = (
    source_loss
    + target_weight * target_loss
    + aux_loss
    + lambda_src_proto * source_proto_loss
    + lambda_tgt_proto * target_proto_loss
)
```

### 关键建议

如果你想让故事更稳，第一版里：

- `source_proto_loss`：对 normal + anomaly 都生效
- `target_proto_loss`：**只对 normal 样本生效**

这样更符合“target 域样本少，重点借 normality 对齐”的叙事。

#### 5）把 prototype 指标写进 metrics
和现在的 `get_moe_metrics()` 一样，把以下指标记到 CSV / log：

- `proto_pull_loss`
- `proto_push_loss`
- `proto_sep_loss`
- `proto_global_normal_dist`
- `proto_global_anomaly_dist`
- `proto_margin_violation`

这些指标以后写 ablation 特别有用。

---

## 5.6 修改 `build_arg_parser()` / 启动脚本

`approaches/MetaLog.py` 和 `MetaLog_BH.py` 只是简单调用 `build_arg_parser()` 和 `run_direction(...)`，因此你真正要改的还是 `supervised_protocol.py` 里的参数解析入口。

### 建议增加的命令行参数

```bash
--use-normality-anchor
--prototype-scale 1.0
--prototype-loss-weight 0.1
--prototype-sep-weight 1e-3
--prototype-margin-global 1.0
--prototype-margin-expert 1.0
--prototype-target-normal-only
--router-use-distance
```

### 第一版默认值建议

```text
use_normality_anchor = True
prototype_scale = 0.5 ~ 1.0
prototype_loss_weight = 0.05 ~ 0.2
prototype_sep_weight = 1e-3
prototype_margin_global = 1.0
prototype_margin_expert = 1.0
prototype_target_normal_only = True
router_use_distance = True
```

---

## 6. 训练流程怎么改（但不改实验设置）

实验还是保留当前两条：

- `hdfs_to_bgl`
- `bgl_to_hdfs`

不额外新造 protocol。

### Warmup 阶段
目标：

- 先把 source 上的 normality anchor 学稳；
- 让 router 学会粗分 normal / boundary / drift 样本。

### Joint 阶段
目标：

- 用少量 target train 样本把 normal prototype 往 target 分布拉；
- 同时保留 source 上的 anomaly discrimination。

---

## 8. 我认为最值得你坚持的表述方式

等你把这套东西接进去以后，论文里不要再把自己写成：

- parser-free + BiMamba + MoE + stage training

而要写成：

> 我们提出一种面向跨系统日志异常检测的 **normality-anchored drift-aware mixture-of-experts**。其核心思想是：在少量 target 监督下，不直接依赖已见异常模式，而是通过可迁移的正常模式边界实现更稳健的跨系统检测。

这个说法会比“我们堆了几个模块”强很多。

---

## 9. 一句话结论

这次最好的改法不是继续缝一个新 backbone，也不是再加一个普通 loss，而是：

1. 用 **prototype / normality anchoring** 把故事立起来；
2. 用 **drift-aware router** 让 MoE 成为为故事服务的机制；
3. 尽量不动实验外框，只在当前 HDFS<->BGL 设定里把方法做扎实。

这条线是当前仓库上**最容易落地、同时最能增强 novel story** 的路线。
