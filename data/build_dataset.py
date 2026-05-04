"""
AgentSheet Dataset Builder
===========================
  python data/build_dataset.py --raw-dir data/raw --out-dir data/processed --stage all
"""

import argparse
import json
import os
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


class FeatureExtractor:
    """
    {
      "sheet_id": "finqa_abc::Sheet1",
      "source": "finqa",
      "num_rows": 10,
      "num_cols": 5,
      "headers": [
        {
          "text": "Revenue",
          "col_idx": 0,
          "dtype": "float64",
          "value_stats": {
            "mean_norm": 0.42,
            "std_norm": 0.18,
            "min_norm": 0.0,
            "max_norm": 1.0,
            "null_ratio": 0.0,
            "is_numeric": 1.0
          }
        },
        ...
      ]
    }
    """

    def extract_from_dataframe(
        self, df: pd.DataFrame, sheet_id: str, source: str = "unknown"
    ) -> Dict:
        num_rows, num_cols = df.shape
        headers = []

        for col_idx, col_name in enumerate(df.columns):
            col_data = df[col_name]
            dtype_str = str(col_data.dtype)
            is_numeric = pd.api.types.is_numeric_dtype(col_data)

            value_stats = {"is_numeric": float(is_numeric), "null_ratio": float(col_data.isna().mean())}
            if is_numeric:
                valid = col_data.dropna()
                if len(valid) > 0:
                    vmin, vmax = float(valid.min()), float(valid.max())
                    rng = vmax - vmin if vmax != vmin else 1.0
                    value_stats.update({
                        "mean_norm": float((valid.mean() - vmin) / rng),
                        "std_norm":  float(valid.std() / rng) if len(valid) > 1 else 0.0,
                        "min_norm":  0.0,
                        "max_norm":  1.0,
                    })
                else:
                    value_stats.update({"mean_norm": 0.0, "std_norm": 0.0, "min_norm": 0.0, "max_norm": 0.0})
            else:
                value_stats.update({"mean_norm": 0.0, "std_norm": 0.0, "min_norm": 0.0, "max_norm": 0.0})

            headers.append({
                "text":       str(col_name).strip(),
                "col_idx":    col_idx,
                "dtype":      dtype_str,
                "value_stats": value_stats,
            })

        return {
            "sheet_id":  sheet_id,
            "source":    source,
            "num_rows":  num_rows,
            "num_cols":  num_cols,
            "headers":   headers,
        }

    def extract_from_json_record(self, record: Dict) -> Optional[Dict]:
        sheet_id = record.get("sheet_id", "")
        headers = record.get("headers", [])
        enriched_headers = []
        for h in headers:
            if isinstance(h, str):
                enriched_headers.append({
                    "text": h, "col_idx": len(enriched_headers), "dtype": "object",
                    "value_stats": {"is_numeric": 0.0, "null_ratio": 0.0,
                                    "mean_norm": 0.0, "std_norm": 0.0,
                                    "min_norm": 0.0, "max_norm": 0.0},
                })
            elif isinstance(h, dict):
                if "value_stats" not in h:
                    h["value_stats"] = {"is_numeric": 0.0, "null_ratio": 0.0,
                                        "mean_norm": 0.0, "std_norm": 0.0,
                                        "min_norm": 0.0, "max_norm": 0.0}
                enriched_headers.append(h)
        record["headers"] = enriched_headers
        return record

    def process_features_file(self, input_path: str, output_path: str):
        with open(input_path) as f:
            records = json.load(f)
        enriched = [self.extract_from_json_record(r) for r in records]
        with open(output_path, "w") as f:
            json.dump(enriched, f, ensure_ascii=False, indent=2)
        print(f"✓ 特征文件已补充 value_stats: {output_path} ({len(enriched)} 条)")


class NwayDataBuilder:
    def __init__(
        self,
        all_sheet_ids: List[str],
        num_distractors: int = 3,
        max_workspace_size: int = 10,
        seed: int = 42,
    ):
        self.all_sheet_ids = all_sheet_ids
        self.num_distractors = num_distractors
        self.max_workspace_size = max_workspace_size
        self.rng = random.Random(seed)

    def build_from_qa_records(self, qa_records: List[Dict]) -> List[Dict]:

        nway_samples = []
        for i, record in enumerate(tqdm(qa_records, desc="构造 N-way 样本")):
            query = record.get("question", record.get("query", ""))
            answer = record.get("answer", "")
            relevant = record.get("highlighted_table", record.get("relevant_subset", []))

            if not query or not relevant:
                continue

            candidate_distractors = [s for s in self.all_sheet_ids if s not in relevant]
            num_dist = min(self.num_distractors, len(candidate_distractors),
                           self.max_workspace_size - len(relevant))
            distractors = self.rng.sample(candidate_distractors, num_dist) if num_dist > 0 else []

            workspace = relevant + distractors
            self.rng.shuffle(workspace)

            nway_samples.append({
                "id": f"nway_{i:06d}",
                "query": query,
                "workspace": workspace,
                "relevant_subset": relevant,
                "answer": answer,
                "source": record.get("source", "unknown"),
            })

        return nway_samples

    def split_and_save(
        self,
        samples: List[Dict],
        out_dir: str,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
    ):
        self.rng.shuffle(samples)
        n = len(samples)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        splits = {
            "train": samples[:n_train],
            "eval":  samples[n_train:n_train + n_val],
            "test":  samples[n_train + n_val:],
        }
        os.makedirs(out_dir, exist_ok=True)
        for split, data in splits.items():
            path = os.path.join(out_dir, f"nway_{split}.json")
            with open(path, "w") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"✓ {split}: {len(data)} 条 → {path}")



class HardNegativeMiner:


    def __init__(self, sbert_model_name: str = "all-MiniLM-L6-v2", batch_size: int = 64):
        self.sbert_model_name = sbert_model_name
        self.batch_size = batch_size
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.sbert_model_name)
                print(f"✓ 加载 SBERT: {self.sbert_model_name}")
            except ImportError:
                raise ImportError("请安装: pip install sentence-transformers")

    def _sheet_to_text(self, feature: Dict) -> str:
        headers = feature.get("headers", [])
        header_texts = [h.get("text", "") if isinstance(h, dict) else str(h) for h in headers[:20]]
        source = feature.get("source", "")
        return f"[{source}] " + " | ".join(header_texts)

    def compute_embeddings(self, features: List[Dict]) -> np.ndarray:
        self._load_model()
        texts = [self._sheet_to_text(f) for f in features]
        embeddings = self._model.encode(
            texts, batch_size=self.batch_size, show_progress_bar=True, normalize_embeddings=True
        )
        return embeddings  # (N, D)

    def mine_hard_negatives(
        self,
        features: List[Dict],
        existing_pairs: List[Dict],
        hard_neg_ratio: float = 0.3,
        sim_low: float = 0.3,
        sim_high: float = 0.6,
    ) -> Tuple[List[Dict], List[Dict]]:

        embeddings = self.compute_embeddings(features)
        id_to_idx = {f["sheet_id"]: i for i, f in enumerate(features)}

        N = len(features)
        sim_matrix = np.zeros((N, N), dtype=np.float32)
        chunk = 512
        for i in range(0, N, chunk):
            sim_matrix[i:i+chunk] = embeddings[i:i+chunk] @ embeddings.T

        existing_set = set()
        for p in existing_pairs:
            a = p.get("sheet_a", p.get("sheet1_id", ""))
            b = p.get("sheet_b", p.get("sheet2_id", ""))
            existing_set.add((min(a, b), max(a, b)))

        hard_neg_pairs = []
        label_corrections = []
        target_hard_neg = int(len(existing_pairs) * hard_neg_ratio)

        for i in range(N):
            for j in range(i + 1, N):
                id_i = features[i]["sheet_id"]
                id_j = features[j]["sheet_id"]
                pair_key = (min(id_i, id_j), max(id_i, id_j))
                sim = float(sim_matrix[i, j])

                if pair_key in existing_set:
                    for p in existing_pairs:
                        a = p.get("sheet_a", p.get("sheet1_id", ""))
                        b = p.get("sheet_b", p.get("sheet2_id", ""))
                        if (min(a, b), max(a, b)) == pair_key and p.get("label", 0) == 0 and sim > 0.7:
                            label_corrections.append({
                                "sheet_a": id_i, "sheet_b": id_j,
                                "current_label": 0, "suggested_label": 1,
                                "similarity": sim,
                            })
                else:
                    # Hard negative：同领域，相似度在 [sim_low, sim_high]
                    src_i = features[i].get("source", "")
                    src_j = features[j].get("source", "")
                    if src_i == src_j and sim_low <= sim <= sim_high:
                        hard_neg_pairs.append({
                            "sheet_a": id_i, "sheet_b": id_j,
                            "label": 0, "pair_type": "hard_negative",
                            "similarity": sim,
                        })
                        if len(hard_neg_pairs) >= target_hard_neg:
                            break
            if len(hard_neg_pairs) >= target_hard_neg:
                break

        print(f"✓ 挖掘到 {len(hard_neg_pairs)} 条 hard negative")
        print(f"✓ 检测到 {len(label_corrections)} 条疑似 false positive")
        return hard_neg_pairs, label_corrections


class PairwiseBuilder:

    def __init__(
        self,
        sim_threshold_pos: float = 0.7,  
        sim_threshold_neg: float = 0.4, 
        seed: int = 42,
    ):
        self.sim_threshold_pos = sim_threshold_pos
        self.sim_threshold_neg = sim_threshold_neg
        self.rng = random.Random(seed)

    def build_augmentation_pairs(
        self,
        features: List[Dict],
        aug_group_key: str = "base_sheet_id",
    ) -> List[Dict]:

        groups = defaultdict(list)
        for f in features:
            base_id = f.get(aug_group_key, f["sheet_id"])
            groups[base_id].append(f["sheet_id"])

        pairs = []
        for base_id, sheet_ids in groups.items():
            if len(sheet_ids) < 2:
                continue
            for i in range(len(sheet_ids)):
                for j in range(i + 1, len(sheet_ids)):
                    pairs.append({
                        "sheet_a": sheet_ids[i],
                        "sheet_b": sheet_ids[j],
                        "label": 1,
                        "pair_type": "augmentation",
                    })
        print(f"✓ Augmentation 正样本: {len(pairs)} 条")
        return pairs

    def build_split_pairs(self, features: List[Dict]) -> List[Dict]:

        groups = defaultdict(list)
        for f in features:
            parent = f.get("parent_sheet_id")
            if parent:
                groups[parent].append(f["sheet_id"])

        pairs = []
        for parent_id, sheet_ids in groups.items():
            if len(sheet_ids) < 2:
                continue
            for i in range(len(sheet_ids)):
                for j in range(i + 1, len(sheet_ids)):
                    pairs.append({
                        "sheet_a": sheet_ids[i],
                        "sheet_b": sheet_ids[j],
                        "label": 1,
                        "pair_type": "split",
                    })
        print(f"✓ Split 正样本: {len(pairs)} 条")
        return pairs

    def build_easy_negatives(
        self, features: List[Dict], n_neg: int, existing_pairs: List[Dict]
    ) -> List[Dict]:
        existing_set = {
            (min(p.get("sheet_a", ""), p.get("sheet_b", "")),
             max(p.get("sheet_a", ""), p.get("sheet_b", "")))
            for p in existing_pairs
        }
        all_ids = [f["sheet_id"] for f in features]
        pairs = []
        attempts = 0
        while len(pairs) < n_neg and attempts < n_neg * 10:
            a, b = self.rng.sample(all_ids, 2)
            key = (min(a, b), max(a, b))
            if key not in existing_set:
                pairs.append({"sheet_a": a, "sheet_b": b, "label": 0, "pair_type": "easy_negative"})
                existing_set.add(key)
            attempts += 1
        print(f"✓ Easy negative: {len(pairs)} 条")
        return pairs

    def save(self, pairs: List[Dict], out_dir: str, train_ratio: float = 0.9):
        self.rng.shuffle(pairs)
        n_train = int(len(pairs) * train_ratio)
        splits = {"pairwise_train": pairs[:n_train], "pairwise_eval": pairs[n_train:]}
        os.makedirs(out_dir, exist_ok=True)
        for name, data in splits.items():
            path = os.path.join(out_dir, f"{name}.json")
            with open(path, "w") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            pos = sum(1 for d in data if d["label"] == 1)
            neg = len(data) - pos
            print(f"✓ {name}: {len(data)} 条 (pos={pos}, neg={neg}) → {path}")


def main():
    parser = argparse.ArgumentParser(description="AgentSheet 数据构造 Pipeline")
    parser.add_argument("--raw-dir", required=True, help="原始数据目录")
    parser.add_argument("--out-dir", required=True, help="输出目录")
    parser.add_argument("--features-file", default=None, help="现有 sheet_features.json 路径")
    parser.add_argument("--qa-file", default=None, help="多表 QA 数据文件路径")
    parser.add_argument("--stage", choices=["features", "pairwise", "nway", "hard_neg", "all"],
                        default="all")
    parser.add_argument("--num-distractors", type=int, default=5)
    parser.add_argument("--hard-neg-ratio", type=float, default=0.3)
    parser.add_argument("--sbert-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    extractor = FeatureExtractor()

    if args.stage in ("features", "all") and args.features_file:
        out_features = os.path.join(args.out_dir, "sheet_features.json")
        extractor.process_features_file(args.features_file, out_features)
        features_path = out_features
    else:
        features_path = args.features_file or os.path.join(args.out_dir, "sheet_features.json")

    if not os.path.exists(features_path):
        print(f"⚠ 特征文件不存在: {features_path}，跳过后续步骤")
        return

    with open(features_path) as f:
        features = json.load(f)
    all_sheet_ids = [f["sheet_id"] for f in features]
    print(f"加载 {len(features)} 条 sheet 特征")

    if args.stage in ("pairwise", "all"):
        builder = PairwiseBuilder(seed=args.seed)
        all_pairs = []
        all_pairs += builder.build_augmentation_pairs(features)
        all_pairs += builder.build_split_pairs(features)
        n_pos = len(all_pairs)
        all_pairs += builder.build_easy_negatives(features, n_pos, all_pairs)
        builder.save(all_pairs, args.out_dir)

    if args.stage in ("nway", "all") and args.qa_file:
        with open(args.qa_file) as f:
            qa_records = json.load(f)
        nway_builder = NwayDataBuilder(
            all_sheet_ids=all_sheet_ids,
            num_distractors=args.num_distractors,
            seed=args.seed,
        )
        samples = nway_builder.build_from_qa_records(qa_records)
        nway_builder.split_and_save(samples, args.out_dir)

    if args.stage in ("hard_neg", "all"):
        pairwise_path = os.path.join(args.out_dir, "pairwise_train.json")
        if os.path.exists(pairwise_path):
            with open(pairwise_path) as f:
                existing_pairs = json.load(f)
            miner = HardNegativeMiner(sbert_model_name=args.sbert_model)
            hard_negs, corrections = miner.mine_hard_negatives(
                features, existing_pairs, hard_neg_ratio=args.hard_neg_ratio
            )
            augmented = existing_pairs + hard_negs
            random.shuffle(augmented)
            out_path = os.path.join(args.out_dir, "pairwise_train.json")
            with open(out_path, "w") as f:
                json.dump(augmented, f, ensure_ascii=False, indent=2)
            print(f"✓ 训练集已增强: {len(augmented)} 条")
            if corrections:
                with open(os.path.join(args.out_dir, "label_corrections.json"), "w") as f:
                    json.dump(corrections, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
