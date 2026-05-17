# HPC 混合数据集改造说明

## 本次完成的改造

### 1. 新增 `HPC` 混合数据集

新数据集目录：

- `datasets/HPC/HPC.log`
- `datasets/HPC/raw_messages.txt`
- `datasets/HPC/raw_log_seqs.txt`
- `datasets/HPC/label.txt`
- `datasets/HPC/block_sources.txt`
- `datasets/HPC/mixture_metadata.json`

这套数据是由以下三个超算数据源混合构成：

- `BGL`
- `TDB(Thunderbird)`
- `Liberty`

### 2. 抽样方式

抽样是按 **chunk / block 级别** 做的，不是按单条日志随机打散。

具体做法：

- 每个源数据集都按节点键 `tokens[3]` 保持原始顺序累积
- 每满 `120` 行形成一个 chunk
- 不足 `120` 行的尾块保留
- 采样时以 chunk 为单位进入混合数据集

这保证了：

- 不破坏单个 chunk 内部的时间顺序
- 不把原始序列打碎成 trace-free 的独立行
- 生成的 `label.txt` 仍然是块级异常检测标签

### 3. 新增代码

- `preprocessing/dataloader/HPCLoader.py`
- `scripts/build_hpc_mixture.py`

### 4. 主流程改造

主训练方向已从原来的：

- `hdfs_to_bgl`
- `bgl_to_hdfs`

改为：

- `hdfs_to_hpc`
- `hpc_to_hdfs`

对应入口：

- `approaches/MetaLog.py`
- `approaches/MetaLog_BH.py`

### 5. 预处理/表示层改造

已支持：

- `Preprocessor.process(dataset='HPC', ...)`
- `cache_parser_free_embeddings.py --dataset HPC`
- `parser_free` 中 `dataset == 'HPC'` 的归一化入口

此外，`HPCLoader` 会优先读取：

- `raw_log_seqs.txt`
- `label.txt`
- `raw_messages.txt`

这意味着后续 parser-free 预处理不会再回扫那两个几十 GB 的解压原始日志。

## 当前数据结果

当前 `datasets/HPC/mixture_metadata.json` 的结果为：

- 总行数：`7,062,509`
- 总块数：`62,502`

各源贡献行数：

- `BGL`: `2,062,349`
- `TDB`: `2,500,080`
- `Liberty`: `2,500,080`

各源块数完全对齐：

- `BGL`: `20,834`
- `TDB`: `20,834`
- `Liberty`: `20,834`

并且三个源都同时包含正常块和异常块：

- 每个源都是 `11,996 Normal chunks + 8,838 Anomalous chunks`

### 与 BGL 的比例对齐情况

当前 `HPC` 的 **chunk 级** 比例：

- `Normal`: `35,988 / 62,502 = 57.5790%`
- `Anomalous`: `26,514 / 62,502 = 42.4210%`

原始完整 `BGL` 的 **chunk 级** 比例：

- `Normal`: `49,274 / 85,577 = 57.5786%`
- `Anomalous`: `36,303 / 85,577 = 42.4214%`

也就是说，这一版 `HPC` 的 **chunk 正常/异常比例已经基本与原始 BGL 对齐**。

### 行级比例说明

当前 `HPC` 的 **行级** 比例：

- `Normal`: `84.3447%`
- `Anomalous`: `15.6553%`

原始完整 `BGL` 的 **行级** 比例：

- `Normal`: `92.6609%`
- `Anomalous`: `7.3391%`

这里和 BGL 不完全一致是正常的，因为我们这次对齐的是 **chunk 级异常检测分布**，不是行级分布。  
对时序异常检测来说，块级分布通常比逐行比例更关键。

## 当前版本的边界

你要求的目标是“总量在 `750w` lines 左右，并且 chunk 比例尽量接近 BGL”。当前版本做到的是：

- chunk 比例已经和 BGL 几乎完全一致
- 三个源都同时有正常/异常 chunk
- 总量达到 `706w` lines

没有到 `750w` 的原因不是脚本没采够，而是：

- `BGL` 在满足当前 chunk 配额后，实际只能提供大约 `206w` 行
- 因为 `BGL` 的平均 chunk 长度显著低于 `120`
- 如果继续强行把总量顶到 `750w`，就必须：
  - 给 `TDB/Liberty` 分配更多 chunk，打破三源块数对齐
  - 或者放宽当前“每源同样 chunk 配额”的设定

所以这版是在“分布更像 BGL”和“总量更接近 750w”之间，优先保证了前者。

## 磁盘清理结果

已删除以下解压后大文件：

- `datasets/TDB/Thunderbird.log`
- `datasets/Liberty/liberty2.log`

当前保留：

- 原始压缩包
- 新的 `datasets/HPC` 混合数据集

## 建议的下一步

下一步建议优先做两件事：

1. 先用 `HPC -> HDFS` 跑一次最小训练烟雾测试，确认训练协议和 checkpoint 目录无新问题
2. 如果你希望总量更接近 `750w`，下一步可以改成“不要求三源 chunk 数完全一致”，让 `TDB/Liberty` 多补一些 chunk
