# AgentSheet 两阶段方案总结（Stage1 + Stage2）

## 1. 总体目标

当前我们把整个系统拆成两阶段：

- **Stage1：Bi-Encoder 表对表示学习 / 相似度学习**
- **Stage2：GTN-based Query-Guided Subgraph Activation**

整体思路是：

1. **Stage1** 先学会把每张 sheet 编成一个向量，并学习 sheet-sheet 的相似关系。
2. **Stage2** 再把一个 workspace 里的多张 sheet 组织成图，在 query 条件下选出 relevant sheets。

---

## 2. Stage1 现在是怎么搭的

### 2.1 Stage1 的定位

Stage1 现在已经**完全换成 Bi-Encoder 路线**，不再使用原来 AgentSheet 的列编码 + cross-sheet attention 结构。

现在 Stage1 的作用是：

- 输入两张表（sheet A, sheet B）
- 分别编码
- 学习它们是否相似 / 是否相关
- 产出可以给 Stage2 用的：
  - sheet embedding
  - pairwise similarity prior

### 2.2 Stage1 输入是什么

每个训练样本是一个 **pairwise 样本**，通常来自：

- `pairwise_train.json`
- `pairwise_eval.json`

每条样本至少需要：

- `sheet_a` / `sheet_b`
  或者直接有
- `sheet1_text` / `sheet2_text`
- `label`（0/1，表示是否相关）

同时依赖 sheet feature 文件，例如：

- `sheet_features_train.json`
- `sheet_features_eval.json`
- 或统一的 `sheet_features.json`

这些 feature 文件里通常会有：

- `sheet_id`
- `source`
- `num_rows`
- `num_cols`
- `headers`

Stage1 会把这些信息序列化成文本，例如：

- source
- shape
- header 列名

然后送进 Bi-Encoder。

### 2.3 Stage1 中间发生了什么

#### (1) 数据层
`SheetSimilarityDataset`

把每张表变成文本表示，例如：

- `source: xxx`
- `shape: 100x12`
- `headers: col1 | col2 | col3 ...`

然后分别 tokenize：

- `text1 -> tokenizer`
- `text2 -> tokenizer`

#### (2) 编码层
`SimilarityClassifier`

它本质上是一个 **共享 backbone 的双塔 / Bi-Encoder**：

- 同一个 Transformer backbone 编码 sheet1
- 同一个 Transformer backbone 编码 sheet2

支持多种 embedding strategy，例如：

- `cls`
- `mean`
- `max`
- `cls_mean_concat`
- `mean_max_concat`
- `cls_mean_max_concat`

也支持：

- layer mix
- extra position embedding
- dropout
- 可选 MLP classifier head

#### (3) 表示组合
得到两个 embedding：

- `emb1`
- `emb2`

然后做经典 Bi-Encoder 组合：

- `u`
- `v`
- `|u-v|`

拼成：

- `[u, v, |u-v|]`

#### (4) 分类头
组合后的向量进入 classifier，输出：

- `logits`

训练时做二分类：

- 相似 / 不相似
- 相关 / 不相关

### 2.4 Stage1 的输出是什么

训练期间主要输出：

- `logits`
- 分类 loss
- accuracy

从模型能力上，它还能提供：

- `sheet embedding`
- `sheet-sheet similarity score`

这些会给 Stage2 用。

### 2.5 Stage1 需要什么数据

至少需要：

```text
data/
  pairwise_train.json
  pairwise_eval.json
  sheet_features_train.json
  sheet_features_eval.json
```

如果没有 split 版 feature，也可以是：

```text
data/
  pairwise_train.json
  pairwise_eval.json
  sheet_features.json
```

### 2.6 Stage1 怎么运行

文件：

- `biencoder_model.py`
- `train_ddp.sh`

运行方式：

```bash
bash train_ddp.sh
```

这是 DDP 版本，会用 `torchrun` 多卡启动。

Stage1 训练完成后，一般会产出：

- `best_model/`
- `final_model/`

后续 Stage2 会优先使用 Stage1 的 backbone / embedding 能力。

---

## 3. Stage2 现在是怎么搭的

### 3.1 Stage2 的定位

Stage2 的目标是：

- 给定一个 query
- 给定一个 workspace 里的 N 张 sheets
- 预测哪些 sheets 是 relevant

和 Stage1 不同：

- Stage1 是 **pairwise**
- Stage2 是 **query + multi-sheet workspace + graph reasoning**

### 3.2 Stage2 输入是什么

每个样本是一个 **N-way / workspace 样本**，通常来自：

- `nway_train.json`
- `nway_eval.json`

每条样本通常包含：

- `query`
- `workspace`：一个 sheet 列表
- `relevant_subset` 或 labels

并且仍然依赖 feature 文件：

- `sheet_features_train.json`
- `sheet_features_eval.json`
- 或 `sheet_features.json`

### 3.3 Stage2 中间发生了什么

Stage2 我们设计了两种模式：

- `gtn_lite`
- `full_gtn`

文件：

- `stage2_gtn.py`

### 3.4 Stage2 的主流程

#### Step 1：编码 query 和 workspace sheets

先用和 Stage1 同类的 backbone：

- 编码 query -> `query_emb`
- 编码每张 sheet -> `sheet_embs`

这里每个节点就是一张 sheet。

#### Step 2：构建多通道图（multi-channel adjacency）

我们不是只建一张图，而是建多张 adjacency matrix。

目前设计的 channel 包括：

##### Channel A：semantic graph
根据 sheet embedding 的相似度构图：

- `A_sem[i, j] = cosine(sheet_i, sheet_j)`

##### Channel B：query-conditioned graph
根据 query 和 sheet 的相关度构图：

- 先算 `s_i = cosine(query, sheet_i)`
- 再构：
  - `A_query[i, j] = s_i * s_j`

意思是：两个都和 query 很相关的节点，应该更容易互相传播信息。

##### Channel C：schema prior graph
根据 header overlap / schema 相似度构图。

##### Channel D：source / shape prior graph
根据 source 一致性、shape 相似性构图。

#### Step 3：GTN 学图

##### 模式 1：`gtn_lite`
做法：

- 给每个 channel 一个可学习权重
- 用 softmax gate 学习如何融合多张图
- 得到一张融合后的图：
  - `A_fused = sum(alpha_k * A_k)`

这个版本更稳、更适合作为第一版。

##### 模式 2：`full_gtn`
做法：

- 学两组 gate：
  - `Q1 = sum(alpha_k * A_k)`
  - `Q2 = sum(beta_k * A_k)`
- 然后做：
  - `A_meta = Q1 @ Q2`

也就是显式学习 meta-path 风格的新图结构。

这个版本更接近 GTN 论文里的图。

#### Step 4：GAT / GNN 消息传播

在 `A_fused` 或 `A_meta` 上做 GAT。

这一层的目标是：

- 让 sheet 节点之间在学到的图上传播信息
- 得到新的节点表示：
  - `node_embs`

同时保留：

- `gat_attn_weights`

这个会给 loss 用。

#### Step 5：query-guided node scoring

最后用 query 去打分每个节点：

- query 和 node embedding 做相似度
- 或用 MLP scorer

输出每个 sheet 的 relevance logits。

### 3.5 Stage2 的 loss 是什么

Stage2 不是单一 loss，而是组合损失。

#### (1) InfoNCE 主损失
让：

- query 更靠近正样本节点
- 远离负样本节点

输入是：

- `query_emb`
- `node_embs`
- `labels`

#### (2) alignment loss
让 GAT 学出的边注意力不要完全乱飞，而是对齐 Stage1 / adjacency prior 的相似度：

- `gat_attn_weights`
- `sheet_sim_scores`

#### (3) subgraph regularization
鼓励正样本节点之间的注意力更大。

也就是：

- relevant sheets 之间形成更强的子图连接

#### (4) BCE / node classification loss
对每个节点做 node-level relevance classification。

### 3.6 Stage2 的输出是什么

训练时主要输出：

- `query_emb`
- `node_embs`
- `gat_attn_weights`
- `sheet_sim_scores`
- `logits`
- 总 loss

推理时最终关心的是：

- 哪些 sheets 被预测为 relevant

### 3.7 Stage2 需要什么数据

至少需要：

```text
data/
  nway_train.json
  nway_eval.json
  sheet_features_train.json
  sheet_features_eval.json
```

或：

```text
data/
  nway_train.json
  nway_eval.json
  sheet_features.json
```

此外，Stage2 通常还会依赖 Stage1 训练得到的 checkpoint，例如：

```text
best_model/
```

用于加载 Stage1 backbone 或 embedding 能力。

### 3.8 Stage2 怎么运行

文件：

- `stage2_gtn.py`
- `train_stage2_ddp.sh`

#### 跑 GTN Lite
```bash
bash train_stage2_ddp.sh gtn_lite
```

#### 跑 Full GTN
```bash
bash train_stage2_ddp.sh full_gtn
```

建议顺序：

1. 先跑 `gtn_lite`
2. 跑通后再跑 `full_gtn`

---

## 4. 两个 Stage 之间的关系

### Stage1 做什么
- 学 sheet 的表示
- 学 sheet-sheet 相似性
- 为 Stage2 提供 embedding / similarity prior

### Stage2 做什么
- 在 query 条件下，把 workspace sheets 组织成图
- 学会 relevant subgraph activation
- 输出最终相关 sheets

所以二者关系可以概括为：

> Stage1 负责学“sheet 长什么样、sheet 和 sheet 之间像不像”；  
> Stage2 负责学“在 query 条件下，workspace 中哪些 sheet 应该被激活”。

---

## 5. 当前代码文件总览

目前 repo 里和这两阶段相关的主要文件是：

### Stage1
- `biencoder_model.py`
- `train_ddp.sh`

### Stage2
- `stage2_gtn.py`
- `train_stage2_ddp.sh`

---

## 6. 推荐给组员的运行顺序

### 第一步：先训练 Stage1
```bash
bash train_ddp.sh
```

### 第二步：确认 Stage1 checkpoint 正常生成
检查是否有：

- `best_model/`
- `final_model/`

### 第三步：先跑 Stage2 Lite
```bash
bash train_stage2_ddp.sh gtn_lite
```

### 第四步：再跑 Full GTN
```bash
bash train_stage2_ddp.sh full_gtn
```

---

## 7. 一句话总结

### Stage1
Bi-Encoder pairwise sheet similarity learning

### Stage2
GTN-based query-guided graph reasoning over workspace sheets

整个系统目标是：

> 先学 sheet 表示，再基于 query 在多图结构上激活 relevant subgraph，最终选出与 query 最相关的 sheets。
