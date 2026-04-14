# Work1 与 MetaLog 对齐改造说明（采样公平版）

## 目标

本说明的目标不是把 `Work1` 改成“更强”或“更像 MetaLog 的全部实现”，而是把 **数据采样协议** 调整到足够接近 MetaLog 仓库的**实际跑法**，从而让两边结果在“sample 层面”更公平可比。

这里要先区分两件事：

1. **sample 公平**：训练/测试切分方式、比例、train 内 anomaly 的保留方式、batch 重采样节奏，对齐 MetaLog 仓库。
2. **full protocol 公平**：除了 sample，还包括 target 模板空间是否提前暴露、是否用 target test 选 best checkpoint 等，也一并对齐。

建议不要覆盖现有 clean 协议，而是新增一个专门的对齐协议。因为当前 `Work1` 默认逻辑本身更干净：

- `hdfs_to_bgl` 使用 `source_ratio=0.3, target_normal_ratio=0.3, target_anomaly_ratio=0.01`
- `bgl_to_hdfs` 使用 `source_ratio=1.0, target_normal_ratio=0.1, target_anomaly_ratio=0.01`
- `prepare_protocol_context()` 里统一走 `split_instances_by_ratio()` 和 `split_instances_by_grouped_label_ratios()`
- target embedding 只保留 `target_train_raw` 中出现过的事件
- target test 中未见事件会 remap 到 OOV

而 MetaLog 仓库的实际跑法不是这套协议：

- `HDFS -> BGL`：`cut_by_316_filter` + `cut_by_415`
- `BGL -> HDFS`：`cut_by_253_filter` + `cut_all`

这些函数都是“先取前缀池，再 shuffle，再切 train/dev/test，并只在 train 内对 anomaly 做 1% 保留”，并不做 sequence-group 隔离。

---

## 当前不公平的根源

### 1. `BGL -> HDFS` 的 target 采样量仍然少于 MetaLog 实跑版

当前 `bgl_to_hdfs` 仍然是：

- `source_ratio=1.0`
- `target_normal_ratio=0.1`
- `target_anomaly_ratio=0.01`

但 MetaLog 实际上在这个方向让 HDFS 走 `cut_by_253_filter`，其逻辑是：

- `dev_split = 0.5 * len(instances)`
- `train_split = 0.2 * len(instances)`
- 用前 `0.7` 的实例池做 shuffle
- 取其中 `20%` 做 train、`50%` 做 dev、后 `30%` 做 test
- 然后只在 train 内对 `Anomalous` 以 `1%` 概率保留

所以在这个方向，**只看 sample，你仍然比 MetaLog 更严格**。

### 2. `HDFS -> BGL` 的 source 采样量仍然少于 MetaLog 实跑版

当前 `hdfs_to_bgl` 是：

- `source_ratio=0.3`
- `target_normal_ratio=0.3`
- `target_anomaly_ratio=0.01`

但 MetaLog 实际上让 HDFS source 走 `cut_by_415`，也就是：

- 前 `0.5` 的实例池参与切分
- `40%` train
- `10%` dev
- `50%` test

所以即便 target headline ratio 看起来接近 MetaLog，**source 侧 sample 仍然没对齐**。

### 3. 你当前按 sequence-group 切，MetaLog 按实例池切

当前 target 的核心切分逻辑是：

- `split_instances_by_grouped_label_ratios()`
- 内部把 normal / anomalous 分开
- 然后分别调用 `split_instances_by_sequence_groups()`

这意味着当前是在**按序列组**做 train/test 隔离。

而 MetaLog 的 `cut_by_316_filter` / `cut_by_253_filter` 只是对前缀实例池 `shuffle` 后直接切，不做 sequence-group 隔离。

因此，哪怕 headline 比例相同，**抽样单位也不一样**，结论仍然不能算“sample 公平”。

---

## 建议的改法

## 总体建议

新增两个协议，而不是覆盖现有实现：

- `protocol=clean`：保留当前协议，作为严谨版主结果。
- `protocol=metalog_repo_sample`：只把 **sample 层** 改到和 MetaLog 实跑版一致。
- `protocol=metalog_repo_full`：在 `metalog_repo_sample` 基础上，再把模板空间暴露与 best-checkpoint 口径也对齐。

建议先做 `metalog_repo_sample`。因为当前最直接的问题是“sample 不公平”，先把这个问题单独解决，后面再决定是否继续对齐到 full protocol。

---

## 第一部分：新增 MetaLog 对齐版切分函数

建议在 `supervised_protocol.py` 里新增一组函数，**不要复用** 当前的 grouped split。

### 1. `cut_all_metalog(instances, rng)`

用于 `BGL -> HDFS` 的 source BGL，对齐 MetaLog 的 `cut_all()`：

```python
def cut_all_metalog(instances, rng):
    shuffled = list(instances)
    rng.shuffle(shuffled)
    return shuffled, [], []
```

### 2. `cut_by_415_metalog(instances, rng)`

用于 `HDFS -> BGL` 的 source HDFS：

```python
def cut_by_415_metalog(instances, rng):
    shuffled = list(instances)
    dev_split = int(0.1 * len(shuffled))
    train_split = int(0.4 * len(shuffled))

    pool = list(shuffled[:train_split + dev_split])
    rng.shuffle(pool)

    train = pool[:train_split]
    dev = pool[train_split:]
    test = shuffled[train_split + dev_split:]
    return train, dev, test
```

### 3. `cut_by_316_filter_metalog(instances, rng)`

用于 `HDFS -> BGL` 的 target BGL：

```python
def cut_by_316_filter_metalog(instances, rng):
    shuffled = list(instances)
    dev_split = int(0.1 * len(shuffled))
    train_split = int(0.3 * len(shuffled))

    pool = list(shuffled[:train_split + dev_split])
    rng.shuffle(pool)

    train = pool[:train_split]
    dev = pool[train_split:]
    test = shuffled[train_split + dev_split:]

    kept = []
    for ins in train:
        if ins.label == 'Anomalous' and rng.rand() > 0.01:
            continue
        kept.append(ins)
    train = kept
    return train, dev, test
```

### 4. `cut_by_253_filter_metalog(instances, rng)`

用于 `BGL -> HDFS` 的 target HDFS：

```python
def cut_by_253_filter_metalog(instances, rng):
    shuffled = list(instances)
    dev_split = int(0.5 * len(shuffled))
    train_split = int(0.2 * len(shuffled))

    pool = list(shuffled[:train_split + dev_split])
    rng.shuffle(pool)

    train = pool[:train_split]
    dev = pool[train_split:]
    test = shuffled[train_split + dev_split:]

    kept = []
    for ins in train:
        if ins.label == 'Anomalous' and rng.rand() > 0.01:
            continue
        kept.append(ins)
    train = kept
    return train, dev, test
```

---

## 第二部分：在 `prepare_protocol_context()` 里分协议分支

当前 `prepare_protocol_context()` 是：

- source 统一走 `split_instances_by_ratio()`
- target 统一走 `split_instances_by_grouped_label_ratios()`

建议改成下面这种结构：

```python
def prepare_protocol_context(direction_key, parser_name, protocol='clean'):
    direction = DIRECTION_CONFIGS[direction_key]
    template_encoder = build_template_encoder(direction.source_dataset)

    source_processor, source_instances = prepare_dataset(...)
    target_processor, target_instances = prepare_dataset(...)

    rng = np.random.RandomState(seed)

    if protocol == 'clean':
        source_train_raw, _ = split_instances_by_ratio(source_instances, direction.source_ratio, rng)
        target_train_raw, target_test_raw = split_instances_by_grouped_label_ratios(
            target_instances,
            direction.target_normal_ratio,
            direction.target_anomaly_ratio,
            rng,
        )
        source_dev_raw = []
        target_dev_raw = []

    elif protocol == 'metalog_repo_sample':
        if direction_key == 'hdfs_to_bgl':
            source_train_raw, source_dev_raw, _ = cut_by_415_metalog(source_instances, rng)
            target_train_raw, target_dev_raw, target_test_raw = cut_by_316_filter_metalog(target_instances, rng)
        elif direction_key == 'bgl_to_hdfs':
            source_train_raw, source_dev_raw, _ = cut_all_metalog(source_instances, rng)
            target_train_raw, target_dev_raw, target_test_raw = cut_by_253_filter_metalog(target_instances, rng)
        else:
            raise ValueError(...)
```

这样做的核心价值是：

- `BGL -> HDFS` 的 target 不再是 `0.1 normal + 0.01 anomaly` 的 clean 版，而是 MetaLog 实跑用的 `cut_by_253_filter` 版。
- `HDFS -> BGL` 的 source 不再是简单 `0.3` 抽样，而是 MetaLog 的 `cut_by_415`。
- target 不再按 sequence-group 切分，而改成 MetaLog 的实例池切分。

---

## 第三部分：sample 对齐后，先保留当前的评估与 vocab 逻辑

如果目标只是 **sample 公平**，建议先**不要动**以下两块：

### 1. 先保留当前的 target embedding 过滤逻辑

当前只用 `source_train_raw` 和 `target_train_raw` 中真正出现过的事件去过滤 embedding，然后构建 merged embeddings；target test 中未见事件会 remap 到 OOV。

这是比 MetaLog 更干净的做法，但它已经不属于“sample 层”的范畴，而是“模板空间暴露口径”的问题。

所以在 `metalog_repo_sample` 阶段，建议先不动这部分，避免一次改太多。

### 2. 先保留当前按 `target_train` 选 best 的逻辑

当前 `evaluate_target()` 是：

- 在 `target_train` 上 tune threshold 或直接算 selection metrics
- 再单独在 `target_test` 上汇报 test metrics

Phase B / Phase C 的 `is_best` 也都是按 `selection_metrics['f']` 决定。

这同样比 MetaLog 更干净。MetaLog 在两个方向上都是直接评估 target test，并按 test F1 存 best model。

这也不属于“sample 层”，所以建议放到下一阶段再决定。

---

## 第四部分：如果要进一步做到“full protocol 公平”

如果后面想让结果和 MetaLog 仓库**完全同口径**，那还要再加一个协议：

### `protocol=metalog_repo_full`

在 `metalog_repo_sample` 基础上再改两件事：

#### 1. target embedding 不再只来自 `target_train_raw`

改成像 MetaLog 一样，直接允许 target 全量 embedding 进入 vocab。

因为 MetaLog 的 `Preprocessor.process()` 之后直接用 `processor_*.embedding` 构造表示，实际等于 target 全量模板空间在切分前已经进入可见词表。

#### 2. best checkpoint 改成按 `target_test` F1 选

即模仿 MetaLog：

- 每个 epoch 直接评估 target test
- 如果 test F1 更高就保存 best model

但不建议把这条作为默认主结果，因为它会让协议不够干净。

---

## 推荐落地顺序

### 第一步：立刻做

新增 `protocol=metalog_repo_sample`，只改 sample：

- 新增 `cut_all_metalog`
- 新增 `cut_by_415_metalog`
- 新增 `cut_by_316_filter_metalog`
- 新增 `cut_by_253_filter_metalog`
- 在 `prepare_protocol_context()` 里按协议分支

这样你就能先回答一个最直接的问题：

> 在 sample 已经对齐 MetaLog 实跑协议后，Work1 还差多少？

### 第二步：实验报告里同时报两组结果

建议最终报告同时给出两列：

- `clean protocol`
- `metalog_repo_sample`

这样能把差距拆开看：

- 如果切到 `metalog_repo_sample` 后分数明显上升，说明以前确实有一部分差距来自 sample 不公平。
- 如果上升不明显，那差距更多来自模型本身或其他非 sample 因素。

### 第三步：除非必须，否则不要直接删掉 clean 协议

当前的 clean 协议至少有三点价值：

1. target 不按 test 泄漏来选模型。
2. target test-only 模板会进 OOV，不会提前暴露。
3. sequence-group 切分更严格，更适合做干净结论。

所以最好的结构不是“把 clean 改坏”，而是：

- **主线保留 clean**
- **增设 metalog_repo_sample 做公平对照**
- **必要时再加 metalog_repo_full 做仓库复现口径**

---

## 最终建议

一句话版建议：

**不要继续微调 `source_ratio` / `target_normal_ratio` 这种表面比例了；直接新增一个 `metalog_repo_sample` 协议，把切分函数和采样单位完整复刻到 MetaLog 仓库的实际跑法。**

因为当前不公平的关键，不只是 `0.1` 和 `0.2` 的差，而是：

- 你按 sequence-group 切
- 它按实例池切
- 你 `HDFS -> BGL` 的 source 还是 `0.3`
- 它实际是 `cut_by_415`
- 你 `BGL -> HDFS` 的 target 还是 `0.1 normal`
- 它实际是 `cut_by_253_filter`

所以，**最优改法不是“改几个数字”，而是“加一个协议层”**。
