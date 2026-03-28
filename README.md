# AgentSheet: Query-Guided Subgraph Activation for Multi-Table Understanding

> NeurIPS 2025 Submission Code

## 项目结构

```
agentsheet_paper/
├── configs/
│   └── default.yaml          # 实验配置（超参数、路径、wandb 等）
├── data/
│   ├── build_dataset.py      # 数据构造 Pipeline
│   │                           (特征提取 / N-way 构造 / Hard Negative 挖掘)
│   ├── raw/                  # 原始数据（FinQA / TableBench / WikiTQ 等）
│   └── processed/            # 处理后的数据（自动生成）
├── models/
│   └── agentsheet.py         # 完整模型实现
│                               (ColumnEncoder / GatedCrossSheetAttention /
│                                SheetGraphBuilder / QueryGuidedActivation)
├── baselines/
│   └── baselines.py          # 三个 Baseline 对比模型
│                               (BM25 / BERT Cross-Encoder / SBERT Bi-Encoder)
├── evaluation/
│   └── evaluate.py           # 评估框架
│                               (Recall@K / MRR / NDCG / Subset EM / 消融实验)
├── scripts/
│   └── train.py              # 完整训练脚本（两阶段 + wandb + checkpoint）
└── experiments/
    └── runs/                 # 实验结果（自动生成）
```

## 快速开始

### 1. 安装依赖

```bash
pip install torch transformers sentence-transformers wandb scikit-learn pyyaml tqdm
```

### 2. 准备数据

```bash
# 补充特征文件（从已有 sheet_features.json 出发）
python data/build_dataset.py \
  --raw-dir data/raw \
  --out-dir data/processed \
  --features-file path/to/sheet_features.json \
  --qa-file path/to/multi_tablebench_qa.json \
  --stage all \
  --num-distractors 5

# 输出：
#   data/processed/sheet_features.json   (补充 value_stats)
#   data/processed/pairwise_train.json   (Stage 1 训练集)
#   data/processed/pairwise_eval.json    (Stage 1 评估集)
#   data/processed/nway_train.json       (Stage 2 训练集)
#   data/processed/nway_eval.json        (Stage 2 评估集)
#   data/processed/nway_test.json        (测试集)
```

### 3. 修改配置

编辑 `configs/default.yaml`，至少修改：
- `model.backbone_name`: BERT 模型路径（本地或 HuggingFace Hub）
- `wandb.entity`: 你的 wandb 用户名
- `wandb.enabled`: 设为 `false` 可关闭 wandb

### 4. 开始训练

```bash
# 完整两阶段训练
python scripts/train.py --config configs/default.yaml

# 只跑 Stage 1（Pairwise 预训练）
python scripts/train.py --config configs/default.yaml --stage 1

# 只跑 Stage 2（N-way 微调，需要先完成 Stage 1）
python scripts/train.py --config configs/default.yaml --stage 2 \
  --resume experiments/runs/agentsheet_full/stage1/best.pt
```

### 5. 评估与对比

```bash
python evaluation/evaluate.py \
  --test-data data/processed/nway_test.json \
  --features data/processed/sheet_features.json
```

## 核心创新点

| 模块 | 创新性 | 对应论文章节 |
|------|--------|------------|
| `ColumnEncoder` | 列级解耦编码：Header 语义 + Value 统计特征门控融合 | §3.1 |
| `GatedCrossSheetAttention` | 无监督列对齐矩阵，可直接可视化 | §3.2 |
| `SheetGraphBuilder` | 完全向量化的 Sheet Graph 构建（O(N²) 矩阵操作） | §3.3 |
| `QueryGuidedActivation` | PPR 风格查询引导激活：不同查询激活不同子图 | §3.4 |

## 消融实验

在 `evaluation/evaluate.py` 中的 `AblationRunner` 自动运行以下消融：

| 消融配置 | 对应 Claim |
|---------|-----------|
| w/o Column-Level Encoding | 证明列级解耦的必要性 |
| w/o Value Stats | 证明统计特征的贡献 |
| w/o Cross-Sheet Attention | 证明列对齐的重要性 |
| w/o Graph Propagation | 证明图结构的必要性 |
| w/o Hard Negative | 证明数据构造策略的影响 |

## 引用

```bibtex
@inproceedings{agentsheet2025,
  title={AgentSheet: Query-Guided Subgraph Activation for Multi-Table Understanding},
  author={...},
  booktitle={Advances in Neural Information Processing Systems},
  year={2025}
}
```
