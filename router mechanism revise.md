# Work1 代码仓库修改建议：从 Global Normality Router 改为 Expert-wise Normality-aware Router

## 0. 结论先行

当前代码已经不是纯粹的普通 MoE，因为 `NormalPrototypeBank` 参与了三件事：

1. `prototype loss`：拉近 normal、推远 anomaly；
2. `prototype-induced logits`：距离 expert prototype 越远，anomaly logit 越大；
3. `router feature`：router 输入里包含到 global prototype 的距离。

但是，当前版本的 normality-aware router 仍然偏弱。原因是 router 只看到一个 `global_distance`，不能判断样本更接近哪一个 expert 的局部正常模式。因此现在更准确的定位是：

> Global-normality-distance-aware Latent MoE

建议改成：

> Expert-wise Normality-Anchored MoE / Prototype-conditioned Normality-aware Router

核心改法是：

```text
删除 global prototype；
每个 expert 只保留一个 normal prototype p_k；
对每个输入表示 h 计算 d_k = distance(h, p_k)；
router 显式参考 [d_1, d_2, ..., d_K] 进行专家选择；
最好进一步把 -alpha * d_k 直接加到 router_logit_k 上。
```

这样，router 的语义就从：

```text
这个样本整体上像不像正常样本？
```

变成：

```text
这个样本更接近哪一个 expert 所锚定的局部正常模式？
```

这更能支撑论文里的 normality-aware router 故事。

---

## 1. 当前代码状态诊断

### 1.1 `models/normality.py`

当前 `NormalPrototypeBank` 同时维护：

```python
self.global_prototype = nn.Parameter(torch.empty(input_dim))
self.expert_prototypes = nn.Parameter(torch.empty(num_experts, input_dim))
```

问题在于：

- `global_prototype` 是随机初始化的可学习向量；
- router 主要使用 `global_distance`；
- expert prototypes 主要参与 prototype loss 和 prototype-induced logits，但没有显式参与 expert 选择。

当前 loss 中包含：

```python
pull_loss = global_distance_sq[normal_mask].mean() + weighted_expert_distance_sq[normal_mask].mean()
```

这会把 normal 同时拉向 global prototype 和 expert prototypes。这个逻辑可以工作，但它会削弱 expert-specific normality 的叙事，因为 global prototype 更像一个总的正常中心，而不是专家级正常模式。

### 1.2 `models/moe.py`

当前 router feature 构造是：

```python
router_feature_dim = input_dim + 2
router_features = concat([normalized_inputs, global_distance, feature_norm])
```

对应 `_build_router_features()`：

```python
return torch.cat([normalized_inputs, global_distance.unsqueeze(-1), feature_norm], dim=-1)
```

这说明 router 只知道一个全局正常性偏离程度，而不知道：

```text
h 到 expert 0 prototype 的距离是多少？
h 到 expert 1 prototype 的距离是多少？
h 到 expert 2 prototype 的距离是多少？
h 到 expert 3 prototype 的距离是多少？
```

因此它还不是强意义上的 expert-wise normality-aware router。

## 2. 推荐目标设计

### 2.1 模型名称建议

建议论文和代码里把模块命名为：

```text
Expert-wise Normality-Anchored MoE
```

或者：

```text
Prototype-conditioned Normality-aware Router
```

它的机制是：

```text
每个 expert k 拥有一个 normal prototype p_k；
输入表示 h 计算到所有 p_k 的距离 d_k；
router 根据 h 和 d_1...d_K 选择 expert；
分类头根据 expert representation 和 expert prototype distance 输出 normal/anomaly logits；
prototype loss 约束 normal 靠近对应 prototype，anomaly 远离 prototype。
```

### 2.2 推荐 router 公式

最推荐的版本是：

```text
base_logit_k = Router_k(h, d_1, ..., d_K)
router_logit_k = base_logit_k - alpha * normalize(d_k)
r_k = softmax(router_logit_k / tau)
```

其中：

- `d_k = ||h - p_k||^2` 或 `sqrt(||h - p_k||^2)`；
- `alpha` 是距离偏置强度，可以先设为超参，也可以做成可学习参数；
- `tau` 是当前已有的 router temperature。

这比单纯 concat 更清楚，因为它直接表达：

> 样本越接近 expert k 的 normal prototype，越倾向路由到 expert k。

---

## 3. 具体代码修改建议

## 3.1 修改 `models/normality.py`

### 目标

把 `NormalPrototypeBank` 从“global + expert prototype bank”改成以 expert prototypes 为主。

建议保留兼容参数：

```python
use_global_prototype=False
```

默认关闭 global prototype，但保留它用于消融实验。

### 建议新增/调整接口

```python
class NormalPrototypeBank(nn.Module):
    def __init__(
        self,
        input_dim,
        num_experts,
        margin_expert=1.0,
        use_global_prototype=False,
        margin_global=1.0,
        diversity_margin=0.5,
        eps=1e-9,
    ):
        ...
```

推荐保留：

```python
self.expert_prototypes = nn.Parameter(torch.empty(num_experts, input_dim))
```

可选保留：

```python
if use_global_prototype:
    self.global_prototype = nn.Parameter(torch.empty(input_dim))
else:
    self.register_parameter('global_prototype', None)
```

### 新增 base representation 到 expert prototypes 的距离

当前 `expert_distance()` 接收的是 `expert_hiddens`，形状大致是 `[B, K, D]`。router 需要在 expert forward 之前就知道 `h` 到每个 prototype 的距离，所以建议新增：

```python
def expert_distance_from_base(self, hiddens):
    # hiddens: [B, D]
    # expert_prototypes: [K, D]
    # return: [B, K]
    diff = hiddens.unsqueeze(1) - self.expert_prototypes.unsqueeze(0)
    return diff.pow(2).mean(dim=-1)
```

保留 expert representation 距离：

```python
def expert_distance_from_expert_repr(self, expert_hiddens):
    # expert_hiddens: [B, K, D]
    # return: [B, K]
    diff = expert_hiddens - self.expert_prototypes.unsqueeze(0)
    return diff.pow(2).mean(dim=-1)
```

### 调整 prototype loss

建议主 loss 使用 expert-only 版本：

```text
L_pull = mean_normal sum_k r_k * ||h_k - p_k||^2
L_push = mean_anomaly sum_k r_k * relu(m - ||h_k - p_k||)^2
```

其中：

- `h_k` 可以使用 expert representation；
- `r_k` 使用 sparse routing probs；
- 如果目标域 anomaly label 不可信或不使用，target 侧继续 `normal_only=True`。

建议不要默认把 normal 拉向 global prototype，因为这会削弱 expert-specific normality。

### 原型分散损失建议

当前 `separation_loss()` 是最小化 pairwise cosine similarity 的平方，这会鼓励 prototype 正交，但不直接保证距离足够分散。建议改成 margin-based distance diversity，或者同时保留 cosine 版本。

推荐新增：

```python
def prototype_diversity_loss(self):
    if self.num_experts < 2:
        return self.expert_prototypes.new_zeros(())
    distances = torch.cdist(self.expert_prototypes, self.expert_prototypes, p=2)
    pair_indices = torch.triu_indices(self.num_experts, self.num_experts, offset=1)
    pairwise = distances[pair_indices[0], pair_indices[1]]
    return F.relu(self.diversity_margin - pairwise).pow(2).mean()
```

这样更能支撑：

> 不同 expert prototype 对应不同局部正常区域。

---

## 3.2 修改 `models/moe.py`

### 新增参数

建议在 `LatentMoEClassifier.__init__()` 增加：

```python
router_distance_mode='expert_bias',
router_distance_scale=1.0,
use_global_prototype=False,
prototype_diversity_margin=0.5,
```

其中 `router_distance_mode` 建议支持：

| 值 | 含义 |
|---|---|
| `none` | 普通 latent MoE，不使用 prototype distance |
| `global_concat` | 当前版本，concat global distance |
| `expert_concat` | concat `[d_1, ..., d_K]` |
| `expert_bias` | `router_logits = base_logits - alpha * d_k` |
| `expert_concat_bias` | 同时 concat distances 并加入 distance bias |

默认建议：

```python
router_distance_mode='expert_bias'
```

或者为了更稳：

```python
router_distance_mode='expert_concat_bias'
```

### 修改 router feature 维度

当前：

```python
self.router_feature_dim = input_dim + (2 if self.router_use_distance else 0)
```

建议改成：

```python
if router_distance_mode in ['expert_concat', 'expert_concat_bias']:
    self.router_feature_dim = input_dim + num_experts + 1  # +1 for feature_norm optional
elif router_distance_mode == 'global_concat':
    self.router_feature_dim = input_dim + 2
else:
    self.router_feature_dim = input_dim
```

其中 `+1` 是 `feature_norm`，可以保留，也可以做成单独开关。

### 在 forward 中先计算 base-to-prototype distances

建议在 `forward()` 开头：

```python
normalized_inputs = self.input_norm(inputs)

if self.prototype_bank is not None:
    expert_base_distances = self.prototype_bank.expert_distance_from_base(inputs)  # [B, K]
else:
    expert_base_distances = None
```

如果需要兼容旧版本：

```python
if use_global_prototype:
    global_distance = self.prototype_bank.global_distance(inputs)
```

### 构造 expert-wise router features

建议 `_build_router_features()` 改成：

```python
def _build_router_features(self, inputs, normalized_inputs, expert_distances=None, global_distance=None):
    if self.router_distance_mode in ['expert_concat', 'expert_concat_bias']:
        feature_norm = torch.norm(inputs, p=2, dim=-1, keepdim=True) / math.sqrt(float(self.input_dim))
        distance_features = torch.log1p(expert_distances)
        return torch.cat([normalized_inputs, distance_features, feature_norm], dim=-1)

    if self.router_distance_mode == 'global_concat':
        feature_norm = torch.norm(inputs, p=2, dim=-1, keepdim=True) / math.sqrt(float(self.input_dim))
        return torch.cat([normalized_inputs, global_distance.unsqueeze(-1), feature_norm], dim=-1)

    return normalized_inputs
```

### 加入 expert-wise distance bias

在得到 `base_router_logits` 后：

```python
router_logits = self.router(router_inputs)

if self.router_distance_mode in ['expert_bias', 'expert_concat_bias']:
    distance_bias = torch.log1p(expert_base_distances)
    distance_bias = distance_bias - distance_bias.mean(dim=-1, keepdim=True)
    distance_bias = distance_bias / (distance_bias.std(dim=-1, keepdim=True) + self.eps)
    router_logits = router_logits - self.router_distance_scale * distance_bias
```

注意：

- 距离需要归一化，否则尺度不稳定；
- `log1p` 可以减少极端距离；
- 减均值除标准差可以让 `alpha` 更好调。

### prototype-induced logits 保持 expert-wise

当前逻辑：

```python
proto_logits = self.prototype_scale * torch.stack([-expert_distances, expert_distances], dim=-1)
expert_logits = expert_logits + proto_logits
```

这个逻辑是好的。建议继续保留，因为它能明确表达：

```text
distance 越大，越偏异常；distance 越小，越偏正常。
```

只需把函数名从当前的 `expert_distance()` 调整为更清楚的：

```python
expert_distances = self.prototype_bank.expert_distance_from_expert_repr(expert_representations)
```

### cache 增加字段

建议 `_last_cache` 增加：

```python
'base_expert_distance': expert_base_distances,
'expert_distance': expert_distances,
'router_logits': router_logits,
```

便于后续 E2 / E3 分析。

### metrics 增加字段

建议 `_last_metrics` 增加：

```python
'proto_base_distance_min': expert_base_distances.min(dim=-1).values.mean().detach(),
'proto_base_distance_mean': expert_base_distances.mean(dim=-1).mean().detach(),
'router_distance_scale': float(self.router_distance_scale),
```

如果 `router_distance_scale` 是可学习参数，则记录其 `.detach()`。

---

## 3.3 修改 `models/mamba.py`

`AttBiMambaModel` 需要透传新增参数：

```python
router_distance_mode='expert_bias',
router_distance_scale=1.0,
use_global_prototype=False,
prototype_diversity_margin=0.5,
```

并传入：

```python
self.proj = LatentMoEClassifier(
    ...,
    router_distance_mode=router_distance_mode,
    router_distance_scale=router_distance_scale,
    use_global_prototype=use_global_prototype,
    prototype_diversity_margin=prototype_diversity_margin,
)
```

日志建议也改一下，不要只打印 `router_use_distance`，而是打印：

```text
router_distance_mode=expert_bias
use_global_prototype=False
prototype_diversity_margin=0.5
```

---

## 10. 最终推荐实现版本

如果只实现一个版本，建议实现：

```text
Expert-only prototypes + expert_bias router
```

即：

```text
删除 router 对 global prototype distance 的依赖；
保留 K 个 expert normal prototypes；
计算 d_k = distance(h, p_k)；
router_logits_k = Router_k(h) - alpha * normalize(d_k)；
expert_logits 继续加入 prototype-induced anomaly logits；
prototype loss 使用 routing-weighted pull-push；
prototype diversity 使用 margin-based distance loss。
```

这条路径代码改动不算大，但能显著增强方法故事的可解释性和可辩护性。
