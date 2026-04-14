# Parser-Free 改造指导文档（轻预处理 + BERT/PLM 语义表示路线）

## 1. 目标

把你当前仓库从“**parser / template / event-id 驱动**”改造成“**raw log text 驱动**”的 parser-free 日志异常检测框架。

这次改造建议走一条**最小侵入、可逐步验收**的路线：

- 不再依赖 Drain / Spell / IPLoM 之类的 log parser
- 不再把模板 ID、事件 ID 作为主输入
- 直接使用原始日志文本（经过轻量规范化）
- 用 **BERT/PLM** 提取日志语义表示

---

## 2. 先定边界：这次改什么，不改什么

### 2.1 这次一定要改的

1. 去掉 parser 依赖
2. 输入单元从 template / event id 改为 raw log message
3. 新增轻量预处理（regex normalization）
4. 新增 tokenizer + PLM encoder
5. 重写或改写 dataset / collate / embedding pipeline

### 2.3 推荐的最小可行目标（MVP）

先做出这个版本：

`normalized raw log -> tokenizer -> frozen BERT -> mean pooling`

---

## 3. 总体改造思路

### 3.1 旧数据流（parser-based）

典型是这样：

`raw log -> parser -> template -> template id / event id -> embedding -> sequence model -> classifier`

### 3.2 新数据流（parser-free）

改造成：

`raw log -> regex normalize -> tokenizer -> PLM(BERT) -> event embedding -> sequence encoder -> classifier`

因此最稳的做法不是把所有东西推倒重来，而是：

> **把“parser + template embedding”这层替换为“normalize + BERT embedding”。**

---

## 4. 推荐的三阶段改造路线

---

### 阶段 A：最小侵入版（强烈推荐先做）

目标：

- 只把输入从 `event_id/template_id` 改成 `BERT event embedding`

架构：

`raw log -> normalize -> BERT -> mean pooling -> 得到每条日志的向量 -> 复用现有序列模型`

适用场景：

- 你现在仓库已经有成熟的 sequence encoder
- 你想先验证 parser-free 是否有效
- 你不想一次性大改

---

## 5. 输入预处理该怎么做

parser-free 不等于完全不预处理。

但这里的预处理目标不是“抽模板”，而是：

- 降噪
- 提高跨系统鲁棒性
- 保留尽可能多的语义结构

### 5.1 不建议的做法

1. **不要像 parser 一样生成模板**
2. 不要强依赖固定位置规则抽字段
3. 不要把一切数字都直接删空
4. 不要把整条日志裁剪得只剩少量关键词

### 5.2 推荐的做法：占位符替换，而不是粗暴删除

推荐把高变化字段替换成特殊 token，而不是直接删除。

例如：

- IP -> `<IP>`
- Port -> `<PORT>`
- 数字 -> `<NUM>`
- 路径 -> `<PATH>`
- UUID -> `<UUID>`
- 十六进制地址 -> `<HEX>`
- block/job/task/container id -> `<ID>`

这样做的好处：

- 保留“这里有一个参数”的结构信息
- 降低跨实例差异
- 比直接删除更稳

### 5.3 推荐的 regex 规范化顺序

建议顺序如下：

1. 转小写（可选，英文日志通常建议）
2. 识别并替换 UUID
3. 替换 IP
4. 替换端口
5. 替换路径
6. 替换十六进制值
7. 替换长数字/普通数字
8. 清理重复空格
9. 保留基本分隔符，或做轻量标点规整

### 5.4 建议保留什么，不建议过度清洗什么

#### 建议保留

- 单词顺序
- 关键动词和组件名
- 部分分隔结构
- error/warn/fail/timeout 等错误触发词
- 节点角色词（namenode, datanode, kernel, executor 等）

#### 谨慎处理

- 全部标点
- 全部数字
- 大小写
- 下划线和中划线

因为很多日志的异常恰恰藏在这些局部结构里。

---

## 6. 推荐的 normalize 示例

### 原始日志

```text
081109 203518 143 INFO dfs.DataNode$DataXceiver: Receiving block blk_-1608999687919862906 src: /10.250.19.102:54106 dest: /10.250.19.102:50010
```

### 建议规范化后

```text
info dfs datanode dataxceiver receiving block <ID> src <IP> <PORT> dest <IP> <PORT>
```

### 再比如

原始：

```text
FATAL RAS KERNEL: instruction cache parity error corrected
```

规范化后：

```text
fatal ras kernel instruction cache parity error corrected
```

注意：

- 第二条几乎不用替换参数，因为它本来就语义很集中
- 第一条中高变化参数被替换成 placeholder，但核心语义还在

---

## 7. 模型层怎么改

这里给你两种推荐方案。

---

### 方案 1：最稳妥方案（优先做）

#### 核心思想

把 BERT 当作“事件级语义编码器”，上层时序模型尽量复用你现有实现。

#### 流程

1. 每条原始日志经过 normalize
2. tokenizer 编码
3. BERT 输出 hidden states
4. 对 token hidden states 做 mean pooling
5. 得到每条日志的 event embedding
6. 将一个session内的 event embedding 序列送入现有序列模型
7. 输出session级异常分数

---

## 8. BERT/PLM 怎么选

### 8.1 首选建议

优先从这两档里选：

#### 稳妥档

- `bert-base-uncased`
- `roberta-base`

---

## 9. pooling 怎么做

### 9.1 推荐默认值

先用 **mean pooling**。

原因：

- 简单稳健
- 日志文本通常短
- 比只取 `[CLS]` 更稳一点
- 与很多日志语义编码做法兼容

### 9.2 可选方案

- `[CLS]`
- attention pooling
- max pooling

但这些都建议放在第二轮实验。

---

### 10.3 如果你当前仓库是 session-based

那也可以保留 session-based，只需要保证：

- session 内每条日志都能变成文本向量
- 上层模型能处理变长序列

---

### 11.3 新增一个 parser-free dataset

文件建议：

```text
data/dataset_parserfree.py
```

建议输出格式：

```python
{
    "raw_logs": [...],
    "norm_logs": [...],
    "label": 0 or 1,
    "meta": {...}
}
```

然后在 collate 里：

- 对一个 batch 内所有日志做 tokenizer
- 编码成 `input_ids / attention_mask`
- reshape 成 `[batch, window, token_len]`

如果你走“离线预提取 embedding”路线，则 dataset 可以直接读缓存后的 `.pt/.npy`。

---

### 11.4 新增 embedding cache 脚本（强烈建议）

文件建议：

```text
scripts/cache_plm_embeddings.py
```

原因：

- BERT 每次在线编码太慢
- 调试时会浪费大量时间
- 你的第一阶段目标是先验证 parser-free，而不是先做最优 end-to-end

建议流程：

1. 先把所有日志规范化
2. 计算每条唯一日志的 embedding
3. 存成 `text -> vector` 缓存

这是最稳的第一步。

---

## 16. 你应该按什么顺序改代码

这是最重要的一节。

### Step 1：切断 parser/template 依赖链

先梳理清楚当前仓库里以下内容在哪：

- parser 输出文件读取
- template vocab 构建
- event id 映射
- embedding lookup

把这条链列出来：

`raw log -> parser output -> template -> event id -> embedding`

然后逐段替换。

---

### Step 2：实现 `LogNormalizer`

先不要碰模型。

只做一件事：

- 输入 raw log
- 输出 normalized text

先抽样检查 100 条日志，看规范化结果是否合理。

验收标准：

- 没有把日志清洗废
- 高变化字段被替换
- 关键语义还在

---

### Step 3：实现 PLM 文本编码脚本

先做单条日志编码测试：

- 输入 10 条日志
- 输出 shape 正确的 embedding
- 检查是否有 nan
- 检查不同日志向量是否并非完全一样

验收标准：

- 编码稳定
- 速度可接受
- 显存/内存可控

---

### Step 4：做 embedding cache

这是第一轮最值得做的工程优化。

做法：

- 对训练集、验证集、测试集里所有唯一 normalized log 去重
- 批量编码
- 保存成缓存字典或数组

推荐保存：

- `id -> text`
- `text -> vector`
- `sample -> sequence of embedding ids`

---

### Step 5：改 dataset / collate

让 dataset 不再返回 `event_id sequence`，而是返回：

- `normalized_text sequence`
- 或 `cached_embedding sequence`

第一轮推荐直接返回 cached embedding。

这样最稳。

---

## 18. 实现上的关键选择：在线编码还是离线编码

### 18.1 第一轮强烈建议：离线编码

也就是：

- 先把每条日志变成 embedding
- 训练时直接读 embedding

优点：

- 快
- 稳
- 容易 debug
- 显存压力低

---

## 21. 常见坑

### 坑 1：把数字全删了

风险：

- 丢掉异常信号
- 例如 error code、端口、节点编号等

建议：

- 优先替换为 placeholder，而不是删空

### 坑 2：normalize 过重

风险：

- 不同日志被洗成几乎一样
- 语义被严重压缩

建议：

- 少做，不够再加，不要一开始做过头

