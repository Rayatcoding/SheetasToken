"""
AgentSheet 评估框架（可运行版）
==============================
保留检索/子集指标计算，并修复原先依赖 model.retrieve 的断裂接口。
当前 evaluate_agentsheet() 直接调用 model(mode='nway', ...)。
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm


class RetrievalMetrics:
    @staticmethod
    def recall_at_k(retrieved: List[List[int]], relevant: List[List[int]], k: int) -> float:
        scores = []
        for ret, rel in zip(retrieved, relevant):
            if not rel:
                continue
            rel_set = set(rel)
            top_k = set(ret[:k])
            scores.append(len(top_k & rel_set) / len(rel_set))
        return float(np.mean(scores)) if scores else 0.0

    @staticmethod
    def mrr(retrieved: List[List[int]], relevant: List[List[int]]) -> float:
        scores = []
        for ret, rel in zip(retrieved, relevant):
            rel_set = set(rel)
            rr = 0.0
            for rank, idx in enumerate(ret, start=1):
                if idx in rel_set:
                    rr = 1.0 / rank
                    break
            scores.append(rr)
        return float(np.mean(scores)) if scores else 0.0

    @staticmethod
    def ndcg_at_k(retrieved: List[List[int]], relevant: List[List[int]], k: int) -> float:
        def dcg(hits, kk):
            return sum(h / np.log2(i + 2) for i, h in enumerate(hits[:kk]))

        scores = []
        for ret, rel in zip(retrieved, relevant):
            rel_set = set(rel)
            hits = [1 if idx in rel_set else 0 for idx in ret]
            ideal = sorted(hits, reverse=True)
            denom = dcg(ideal, k)
            numer = dcg(hits, k)
            scores.append(numer / denom if denom > 0 else 0.0)
        return float(np.mean(scores)) if scores else 0.0

    @staticmethod
    def precision_at_k(retrieved: List[List[int]], relevant: List[List[int]], k: int) -> float:
        scores = []
        for ret, rel in zip(retrieved, relevant):
            rel_set = set(rel)
            top_k = ret[:k]
            scores.append(sum(1 for idx in top_k if idx in rel_set) / max(1, k))
        return float(np.mean(scores)) if scores else 0.0

    @classmethod
    def compute_all(cls, retrieved: List[List[int]], relevant: List[List[int]], ks: List[int] = [1, 3, 5]) -> Dict[str, float]:
        results = {"MRR": cls.mrr(retrieved, relevant)}
        for k in ks:
            results[f"Recall@{k}"] = cls.recall_at_k(retrieved, relevant, k)
            results[f"NDCG@{k}"] = cls.ndcg_at_k(retrieved, relevant, k)
            results[f"P@{k}"] = cls.precision_at_k(retrieved, relevant, k)
        return results


class SubsetMetrics:
    @staticmethod
    def subset_exact_match(predicted: List[List[int]], relevant: List[List[int]]) -> float:
        vals = [1.0 if set(p) == set(r) else 0.0 for p, r in zip(predicted, relevant)]
        return float(np.mean(vals)) if vals else 0.0

    @staticmethod
    def subset_f1(predicted: List[List[int]], relevant: List[List[int]]) -> float:
        f1s = []
        for p, r in zip(predicted, relevant):
            p_set, r_set = set(p), set(r)
            if not p_set and not r_set:
                f1s.append(1.0)
                continue
            if not p_set or not r_set:
                f1s.append(0.0)
                continue
            prec = len(p_set & r_set) / len(p_set)
            rec = len(p_set & r_set) / len(r_set)
            f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0)
        return float(np.mean(f1s)) if f1s else 0.0

    @staticmethod
    def jaccard(predicted: List[List[int]], relevant: List[List[int]]) -> float:
        vals = []
        for p, r in zip(predicted, relevant):
            p_set, r_set = set(p), set(r)
            union = p_set | r_set
            vals.append(len(p_set & r_set) / len(union) if union else 1.0)
        return float(np.mean(vals)) if vals else 0.0

    @classmethod
    def compute_all(cls, predicted: List[List[int]], relevant: List[List[int]]) -> Dict[str, float]:
        return {
            "Subset_EM": cls.subset_exact_match(predicted, relevant),
            "Subset_F1": cls.subset_f1(predicted, relevant),
            "Jaccard": cls.jaccard(predicted, relevant),
        }


class Evaluator:
    def __init__(self, test_data: List[Dict], features_map: Dict[str, Dict], device: torch.device | None = None):
        self.test_data = test_data
        self.features_map = features_map
        self.device = device or torch.device("cpu")

    @staticmethod
    def _relevant_indices(workspace: List[str], relevant_subset: List[str]) -> List[int]:
        rel_set = set(relevant_subset)
        return [i for i, sid in enumerate(workspace) if sid in rel_set]

    def evaluate_agentsheet_from_logits(
        self,
        logits_list: List[List[float]],
        workspaces: List[List[str]],
        relevant_subsets: List[List[str]],
        threshold: float = 0.5,
        ks: List[int] = [1, 3, 5],
    ) -> Dict[str, float]:
        all_retrieved = []
        all_predicted = []
        all_relevant = []

        for scores, workspace, relevant_subset in zip(logits_list, workspaces, relevant_subsets):
            rel_indices = self._relevant_indices(workspace, relevant_subset)
            ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            predicted = [i for i, s in enumerate(scores) if s > threshold]
            if not predicted and ranked:
                predicted = ranked[:1]
            all_retrieved.append(ranked)
            all_predicted.append(predicted)
            all_relevant.append(rel_indices)

        retrieval_metrics = RetrievalMetrics.compute_all(all_retrieved, all_relevant, ks=ks)
        subset_metrics = SubsetMetrics.compute_all(all_predicted, all_relevant)
        return {**retrieval_metrics, **subset_metrics}

    @staticmethod
    def print_table(results_dict: Dict[str, Dict[str, float]]):
        models = list(results_dict.keys())
        metrics = list(next(iter(results_dict.values())).keys())
        col_w = max(len(m) for m in models) + 2
        metric_w = 12
        header = f"{'Model':<{col_w}}" + "".join(f"{m:>{metric_w}}" for m in metrics)
        print("\n" + "=" * len(header))
        print(header)
        print("-" * len(header))
        for model_name, scores in results_dict.items():
            row = f"{model_name:<{col_w}}"
            for m in metrics:
                row += f"{scores.get(m, 0.0) * 100:>{metric_w - 1}.1f}%"
            print(row)
        print("=" * len(header) + "\n")

    @staticmethod
    def save_results(results_dict: Dict[str, Dict[str, float]], out_path: str):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results_dict, f, indent=2)
        print(f"✓ 结果已保存: {out_path}")


def plot_alignment_matrix(
    align_matrix: np.ndarray,
    headers_a: List[str],
    headers_b: List[str],
    title: str = "Column Alignment Matrix",
    save_path: Optional[str] = None,
):
    fig, ax = plt.subplots(figsize=(max(6, len(headers_b) * 0.8), max(4, len(headers_a) * 0.6)))
    im = ax.imshow(align_matrix, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Alignment Score")
    ax.set_xticks(range(len(headers_b)))
    ax.set_xticklabels(headers_b, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(headers_a)))
    ax.set_yticklabels(headers_a, fontsize=9)
    for i in range(len(headers_a)):
        for j in range(len(headers_b)):
            val = align_matrix[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Sheet B Columns")
    ax.set_ylabel("Sheet A Columns")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✓ 对齐矩阵图已保存: {save_path}")
    return fig


def plot_activation_propagation(
    history: List[np.ndarray],
    sheet_ids: List[str],
    relevant_indices: List[int],
    query: str,
    save_path: Optional[str] = None,
):
    num_layers = len(history)
    num_nodes = len(sheet_ids)
    short_ids = [sid.split("::")[-1][:15] for sid in sheet_ids]
    fig, axes = plt.subplots(1, num_layers, figsize=(num_layers * 3, max(4, num_nodes * 0.5)), sharey=True)
    if num_layers == 1:
        axes = [axes]
    colors = ["#d62728" if i in relevant_indices else "#1f77b4" for i in range(num_nodes)]
    for layer_idx, (ax, scores) in enumerate(zip(axes, history)):
        ax.barh(range(num_nodes), scores, color=colors)
        ax.set_xlim(0, 1)
        ax.set_title(f"Layer {layer_idx}" if layer_idx > 0 else "Initial (s⁰)", fontsize=10)
        ax.set_yticks(range(num_nodes))
        if layer_idx == 0:
            ax.set_yticklabels(short_ids, fontsize=8)
        ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✓ 激活传播图已保存: {save_path}")
    return fig
