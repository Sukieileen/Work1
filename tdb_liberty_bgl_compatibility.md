# TDB / Liberty 与 BGL 兼容性结论

## 1. 本次已完成的事情

- 已解压 `datasets/TDB/Thunderbird.tar.gz` 到 `datasets/TDB/Thunderbird.log`
- 已解压 `datasets/Liberty/liberty2 (1).gz` 到 `datasets/Liberty/liberty2.log`

当前磁盘上的主要文件为：

- `datasets/TDB/Thunderbird.log`
- `datasets/Liberty/liberty2.log`

## 2. 先说结论

结论分两层：

1. `Thunderbird` 和 `Liberty` 不能 **直接原封不动** 按当前 `BGL` 流程处理。
2. 但它们和 `BGL` 属于非常接近的 HPC 日志格式，经过 **小到中等规模适配** 后，可以接入当前 parser-free 流程；进一步改训练协议后，也可以做成“超算侧 = `BGL + Thunderbird + Liberty` 的三数据集混合 source domain”。

也就是说：

- **现在不能直接做到**
- **改代码后可以做到**

## 3. 为什么说它们和 BGL 很接近

这三个数据集都有几个明显共同点：

- 每行都是原始日志文本
- 第 1 列都能体现标签信息：`-` 表示正常，非 `-` 表示异常
- 第 4 列都可作为节点/主机维度分组键
- 都适合先切成 session / block，再走 parser-free 事件归一化与编码

样例对比：

`BGL`:

```text
- 1117838570 2005.06.03 R02-M1-N0-C:J12-U11 2005-06-03-15.42.50.363779 R02-M1-N0-C:J12-U11 RAS KERNEL INFO instruction cache parity error corrected
```

`Thunderbird`:

```text
- 1131523501 2005.11.09 aadmin1 Nov 10 00:05:01 src@aadmin1 in.tftpd[14620]: tftp: client does not accept options
```

`Liberty`:

```text
- 1102911148 2004.12.12 ladmin2 Dec 12 20:12:28 src@ladmin2 .... IRQ redirection table:
```

所以从“是否还能沿用 BGL 那种 node 分组 + fixed window 切块 + 行首标签聚合”的角度看，答案是 **可以沿用大框架**。

## 4. 为什么当前不能直接按 BGL 代码处理

### 4.1 预处理入口只支持 `HDFS` 和 `BGL`

当前 `Preprocessor.process()` 里只保留了两个主数据集分支：

- `preprocessing/Preprocess.py:64`
- `preprocessing/Preprocess.py:67`
- `preprocessing/Preprocess.py:73`

也就是：

- `HDFS` 走 `HDFSLoader`
- `BGL/BGLSample` 走 `BGLLoader`
- 其他名字会直接抛 `Unsupported dataset`

因此现在传入 `TDB`、`Thunderbird`、`Liberty` 都会直接失败。

### 4.2 `BGLLoader` 对列结构有硬编码，Thunderbird/Liberty 不能直接复用

当前 `BGLLoader` 里有三个关键硬编码：

- `remove_cols = [0,1,2,3,4,5,6,7,8]`
- `node = tokens[3]`
- `if not line.startswith('-'): label = 'Anomalous'`

对应位置：

- `preprocessing/dataloader/BGLLoader.py:40`
- `preprocessing/dataloader/BGLLoader.py:85`
- `preprocessing/dataloader/BGLLoader.py:100`

其中后两项对 `Thunderbird/Liberty` 基本仍然成立：

- `tokens[3]` 仍然是主机名/节点名
- 行首仍然可判别正常/异常

真正不兼容的是第一项：`remove_cols = [0..8]`。

`BGL` 的消息正文从第 10 列开始，去掉前 9 列是合理的；但 `Thunderbird/Liberty` 的头部更接近 8 列，正文通常从第 9 列开始。  
如果继续按 BGL 逻辑去掉前 9 列，会把正文第一词一起删掉。

更严重的是，`Liberty` 有一部分短日志本身只有 8 列。对前 20 万行样本做检查：

- `BGL`: `empty_remove9 = 0`
- `Thunderbird`: `empty_remove9 = 0`
- `Liberty`: `empty_remove9 = 487`

这表示如果直接复用 `BGLLoader._pre_process()`：

- `Thunderbird` 会出现“正文被截掉首词”的系统性信息损失
- `Liberty` 还会出现一部分日志被裁成空消息

因此 **不能原样复用 `BGLLoader`**。

### 4.3 parser-free 归一化目前只给 `BGL` 做了超算特化

当前 `LogNormalizer` 里只有：

- `if self.dataset == 'BGL': text = self._normalize_bgl(text)`

位置在：

- `representations/parser_free.py:113`

这不是“会直接报错”的问题，但意味着：

- `Thunderbird/Liberty` 即使勉强接进来，也拿不到和 `BGL` 同级别的 HPC 专用归一化
- 主机名、硬件位置、十六进制串、syslog 风格 token 的归一化策略不完全匹配

所以如果要把三个超算数据集真的混合作为同一个 source domain，最好补上针对 `Thunderbird/Liberty` 的 normalizer 规则，或者把 BGL 的 HPC 归一化规则泛化成一套 `HPC` 规则。

## 5. 当前能不能把“超算一侧”直接改成 3 个数据集混合？

### 5.1 现在不能直接做到

原因不是一个，而是三层同时限制：

1. 数据加载层只认 `HDFS/BGL`
2. `BGLLoader` 的列裁剪逻辑不适用于 `Thunderbird/Liberty`
3. 双向训练协议当前是严格的二域设定，不是多 source domain 设定

### 5.2 训练协议为什么也不支持

当前方向配置是硬编码二选一：

- `hdfs_to_bgl`
- `bgl_to_hdfs`

位置在：

- `approaches/supervised_protocol.py:40`
- `approaches/supervised_protocol.py:46`

而 `DirectionConfig` 只有：

- `source_dataset: str`
- `target_dataset: str`

不是 `list[str]`。

后面 `prepare_protocol_context()` 也是按“一个 source + 一个 target”写死的：

- 各自单独建 semantic encoder
- 各自单独 `prepare_dataset()`
- `merged_embeddings` 只合并这两个域
- `source_train` 也只来自单个 source dataset

对应位置：

- `approaches/supervised_protocol.py:661`
- `approaches/supervised_protocol.py:666`
- `approaches/supervised_protocol.py:671`
- `approaches/supervised_protocol.py:699`

所以“把超算一侧改成 3 个数据集混合”不是改个配置就能开，必须改训练上下文构建逻辑。

## 6. 如果要做到，最少需要改哪些地方

我认为是 **可以做到的**，而且不需要推翻现有框架，但至少要做下面几项改造。

### 6.1 新增或泛化 HPC dataloader

建议不要继续叫 `BGLLoader`，而是改成更泛化的 HPC loader，至少参数化这些内容：

- 原始日志路径
- `remove_cols`
- `group_key_index`
- `normal_prefix` 规则
- `window_size`

最低可行方案：

- 保留 `BGLLoader`
- 新增 `ThunderbirdLoader`
- 新增 `LibertyLoader`

其中：

- 三者都可沿用 `tokens[3]` 分组
- 三者都可沿用 `line.startswith('-')` 判正常
- `Thunderbird/Liberty` 需要把消息头裁剪改成更合适的列数，不能继续固定 `[0..8]`

### 6.2 打开 `Preprocessor.process()` 对新数据集的支持

需要在 `preprocessing/Preprocess.py` 增加对应分支，例如：

- `dataset == 'Thunderbird'`
- `dataset == 'Liberty'`

或统一成一个 `HPC` 家族分发逻辑。

### 6.3 让 normalizer 支持更多 HPC 数据

建议把现在的 `_normalize_bgl()` 从“只给 BGL 用”改成“面向 HPC 数据集的通用归一化”，至少考虑：

- 主机名 / 节点名
- syslog 时间头
- 内核模块名
- 十六进制串
- 文件路径
- 进程号 / 设备号 / 错误码

否则三库混合后，模板空间会被无谓放大，source 侧分布会更碎。

### 6.4 把二域协议改成多 source、一 target

这是训练层最关键的一步。  
建议把：

- `source_dataset: str`

改成：

- `source_datasets: list[str]`

然后：

1. 分别预处理 `BGL / Thunderbird / Liberty`
2. 分别得到各自的 `embedding` 与 `instances`
3. 合并三者的 source instances
4. 在 `build_merged_embeddings()` 里把三份 source embedding 和 target embedding 一起 remap

也就是说最终应变成：

- `source = concat(BGL, Thunderbird, Liberty)`
- `target = HDFS`

如果你们说的“双向训练模型时将超算一侧改为 3 个数据集的混合数据集”指的是这个方向，那么技术上是可实现的。

## 7. 我的最终判断

最终判断如下：

- `Thunderbird` 和 `Liberty` 与 `BGL` 高度同源，适合纳入“超算侧”
- 但它们 **不能直接像当前 BGL 一样无修改处理**
- 主要阻塞点在于：
  - loader 只支持 `HDFS/BGL`
  - `BGLLoader` 对消息头裁剪写死为去掉前 9 列
  - 训练协议只支持单 source + 单 target
- 如果愿意做一轮适配，完全可以把超算 source 扩成 `BGL + Thunderbird + Liberty`

一句话总结：

> 现在不能直接混；补上 HPC 数据加载适配和多 source 训练上下文后，可以混，而且从数据形态上看是合理的。

## 8. 建议的下一步

如果你要我继续推进，建议按这个顺序做：

1. 先实现 `Thunderbird/Liberty` 的 loader 适配与 parser-free 预处理落盘
2. 抽样检查三库切块后的标签分布、块长分布、模板数规模
3. 再改 `supervised_protocol.py`，把 `BGL -> HDFS` 扩成 `HPC(=BGL+Thunderbird+Liberty) -> HDFS`

这样风险最低，也最容易定位问题是在“数据切块”还是“训练协议”这一层。
