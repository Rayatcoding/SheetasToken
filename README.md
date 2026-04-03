# AgentSheet (Current Working Version)

> Current repo status after the Stage1 / Stage2 redesign

This repository is no longer using the original paper-style implementation as the main training path.  
The **current working pipeline** is:

- **Stage1:** Bi-Encoder sheet-pair representation learning
- **Stage2:** GTN-based query-guided graph reasoning (`gtn_lite` / `full_gtn`)

The old `models/agentsheet.py` / `scripts/train.py` path is retained only as historical reference.  
The files you should use now are:

- `biencoder_model.py`
- `train_ddp.sh`
- `stage2_gtn.py`
- `train_stage2_ddp.sh`

---

## 1. Current two-stage pipeline

### Stage1: Bi-Encoder sheet similarity learning

**Goal**

Learn a good embedding for each sheet and a pairwise similarity prior between sheets.

**Input**

Training files under `data/`:

- `pairwise_train.json`
- `pairwise_eval.json`

Feature files:

- `sheet_features_train.json`
- `sheet_features_eval.json`

or a shared:

- `sheet_features.json`

Each pairwise sample should contain either:

- `sheet_a`, `sheet_b`, `label`

or directly:

- `sheet1_text`, `sheet2_text`, `label`

**Architecture**

`biencoder_model.py`

1. Convert each sheet into text using feature fields such as:
   - `source`
   - `shape` (`num_rows x num_cols`)
   - `headers`
2. Encode the two sheets with a **shared Transformer backbone**
3. Build pair representation from:
   - `u`
   - `v`
   - `|u-v|`
4. Feed the concatenated representation into a classifier head

**Output**

- pairwise logits
- sheet embeddings
- a Stage1 checkpoint for Stage2 initialization

**Checkpoint location**

The current Stage1 training path saves checkpoints to:

```bash
outputs/stage1_biencoder/best_model/classifier.pt
outputs/stage1_biencoder/final_model/classifier.pt
```

---

### Stage2: GTN-based query-guided subgraph activation

**Goal**

Given a query and a workspace of multiple sheets, predict which sheets are relevant.

**Input**

Training files under `data/`:

- `nway_train.json`
- `nway_eval.json`

Feature files:

- `sheet_features_train.json`
- `sheet_features_eval.json`

or a shared:

- `sheet_features.json`

Each N-way sample should contain:

- `query`
- `workspace` (list of sheet IDs or sheet references)
- relevant labels / relevant subset

**Architecture**

`stage2_gtn.py`

#### Step 1: Encode query and sheets
- query -> `query_emb`
- each workspace sheet -> initial node embedding

#### Step 2: Build multi-channel graph
Typical adjacency channels include:
- semantic graph
- query-conditioned graph
- schema prior graph
- source / shape prior graph

#### Step 3: Learn graph structure
Two supported modes:

**`gtn_lite`**
- learn a softmax gate over graph channels
- fuse them into one graph

**`full_gtn`**
- learn two gated graph mixtures
- multiply them to produce a meta-path-style graph

#### Step 4: Graph propagation
- run GAT on the learned graph
- produce refined node embeddings and attention weights

#### Step 5: Query-guided node scoring
- score each sheet node against the query
- output node relevance logits

**Losses**

Stage2 combines:
- InfoNCE loss
- alignment loss
- subgraph regularization
- BCE node classification loss

**Output**

- `query_emb`
- `node_embs`
- `gat_attn_weights`
- `sheet_sim_scores`
- node logits
- relevant sheet predictions

---

## 2. Repo files to use now

### Main Stage1 files
- `biencoder_model.py`
- `train_ddp.sh`

### Main Stage2 files
- `stage2_gtn.py`
- `train_stage2_ddp.sh`

### Data construction / legacy files
- `data/build_dataset.py`
- `models/agentsheet.py`
- `scripts/train.py`

These legacy files are not the main training entry points anymore.

---

## 3. Data requirements

### Stage1
Required:

```text
data/
  pairwise_train.json
  pairwise_eval.json
  sheet_features_train.json
  sheet_features_eval.json
```

Alternative:

```text
data/
  pairwise_train.json
  pairwise_eval.json
  sheet_features.json
```

### Stage2
Required:

```text
data/
  nway_train.json
  nway_eval.json
  sheet_features_train.json
  sheet_features_eval.json
```

Alternative:

```text
data/
  nway_train.json
  nway_eval.json
  sheet_features.json
```

---

## 4. How to run

### Stage1

```bash
bash train_ddp.sh
```

This launches DDP training for the Bi-Encoder Stage1.

Expected output checkpoint:

```bash
outputs/stage1_biencoder/best_model/classifier.pt
```

---

### Stage2 Lite

```bash
bash train_stage2_ddp.sh gtn_lite
```

### Stage2 Full GTN

```bash
bash train_stage2_ddp.sh full_gtn
```

By default, `train_stage2_ddp.sh` will first try to load:

```bash
outputs/stage1_biencoder/best_model/classifier.pt
```

and fall back to:

```bash
best_model/classifier.pt
```

You can also override manually:

```bash
STAGE1_CKPT=/path/to/classifier.pt bash train_stage2_ddp.sh gtn_lite
```

---

## 5. Recommended run order

### Step 1
Run Stage1 first:

```bash
bash train_ddp.sh
```

### Step 2
Confirm the checkpoint exists:

```bash
outputs/stage1_biencoder/best_model/classifier.pt
```

### Step 3
Run Stage2 Lite first:

```bash
bash train_stage2_ddp.sh gtn_lite
```

### Step 4
Then run Full GTN:

```bash
bash train_stage2_ddp.sh full_gtn
```

---

## 6. Practical notes

1. If Stage2 cannot start, first check whether `nway_train.json` and `nway_eval.json` exist.
2. If Stage2 cannot find the Stage1 checkpoint, either:
   - make sure Stage1 finished normally, or
   - pass `STAGE1_CKPT=/your/path/classifier.pt`
3. If your local backbone path differs, set `MODEL_NAME=/your/model/path`.

---

## 7. One-line summary

- **Stage1** learns sheet embeddings and sheet-sheet similarity
- **Stage2** performs query-guided graph reasoning over a workspace of sheets

In short:

> Stage1 learns what sheets look like and how similar they are;  
> Stage2 learns which subset of sheets should be activated for a given query.
