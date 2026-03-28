"""
AgentSheet 训练脚本（可运行版）
================================
目标：
  - 优先保证按当前数据格式可直接运行
  - Stage 1 可单独训练
  - features 支持单文件或多文件自动合并
  - Stage 2 在数据缺失时自动跳过
  - GPU/CPU 自动适配，服务器可直接用 CUDA

用法：
  python scripts/train.py --config configs/default.yaml --stage 1
  python scripts/train.py --config configs/default.yaml
  python scripts/train.py --config configs/default.yaml --resume path/to/ckpt.pt
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup
except ImportError as e:
    raise ImportError(
        "transformers 未安装。请先在服务器环境中安装 transformers。"
    ) from e

# 将项目根目录加入 path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.agentsheet import AgentSheetConfig, AgentSheetModel
from evaluation.evaluate import RetrievalMetrics, SubsetMetrics


# =============================================================================
# 数据集类
# =============================================================================

class PairwiseDataset(Dataset):
    """Stage 1 Pairwise 数据集"""

    def __init__(
        self,
        pairs: List[Dict],
        features_map: Dict[str, Dict],
        tokenizer,
        max_cols: int = 16,
        max_col_length: int = 32,
        value_dim: int = 6,
    ):
        self.pairs = pairs
        self.features_map = features_map
        self.tokenizer = tokenizer
        self.max_cols = max_cols
        self.max_col_length = max_col_length
        self.value_dim = value_dim
        self.missing_sheet_ids = set()

    def _encode_sheet(self, sheet_id: str) -> Tuple[torch.Tensor, torch.Tensor]:
        feature = self.features_map.get(sheet_id)
        if feature is None:
            self.missing_sheet_ids.add(sheet_id)
            feature = {"headers": []}

        headers = feature.get("headers", [])[:self.max_cols]
        header_ids = torch.zeros(self.max_cols, self.max_col_length, dtype=torch.long)
        value_stats = torch.zeros(self.max_cols, self.value_dim, dtype=torch.float)

        for i, h in enumerate(headers):
            text = h.get("text", "") if isinstance(h, dict) else str(h)
            enc = self.tokenizer(
                text,
                max_length=self.max_col_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            header_ids[i] = enc["input_ids"][0]

            if isinstance(h, dict) and "value_stats" in h:
                vs = h["value_stats"]
                value_stats[i] = torch.tensor([
                    float(vs.get("mean_norm", 0.0)),
                    float(vs.get("std_norm", 0.0)),
                    float(vs.get("min_norm", 0.0)),
                    float(vs.get("max_norm", 0.0)),
                    float(vs.get("null_ratio", 0.0)),
                    float(vs.get("is_numeric", 0.0)),
                ], dtype=torch.float)

        return header_ids, value_stats

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        sheet_a = pair.get("sheet_a", pair.get("sheet1_id", ""))
        sheet_b = pair.get("sheet_b", pair.get("sheet2_id", ""))
        label = float(pair.get("label", 0.0))

        ids_a, stats_a = self._encode_sheet(sheet_a)
        ids_b, stats_b = self._encode_sheet(sheet_b)

        return {
            "header_ids_a": ids_a,
            "value_stats_a": stats_a,
            "header_ids_b": ids_b,
            "value_stats_b": stats_b,
            "label": torch.tensor(label, dtype=torch.float),
        }


class NwayDataset(Dataset):
    """Stage 2 N-way 数据集"""

    def __init__(
        self,
        samples: List[Dict],
        features_map: Dict[str, Dict],
        tokenizer,
        max_workspace_size: int = 10,
        max_cols: int = 16,
        max_col_length: int = 32,
        max_query_length: int = 64,
        value_dim: int = 6,
    ):
        self.samples = samples
        self.features_map = features_map
        self.tokenizer = tokenizer
        self.max_workspace_size = max_workspace_size
        self.max_cols = max_cols
        self.max_col_length = max_col_length
        self.max_query_length = max_query_length
        self.value_dim = value_dim
        self.missing_sheet_ids = set()

    def _encode_sheet(self, sheet_id: str) -> Tuple[torch.Tensor, torch.Tensor]:
        feature = self.features_map.get(sheet_id)
        if feature is None:
            self.missing_sheet_ids.add(sheet_id)
            feature = {"headers": []}

        headers = feature.get("headers", [])[:self.max_cols]
        header_ids = torch.zeros(self.max_cols, self.max_col_length, dtype=torch.long)
        value_stats = torch.zeros(self.max_cols, self.value_dim, dtype=torch.float)

        for i, h in enumerate(headers):
            text = h.get("text", "") if isinstance(h, dict) else str(h)
            enc = self.tokenizer(
                text,
                max_length=self.max_col_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            header_ids[i] = enc["input_ids"][0]
            if isinstance(h, dict) and "value_stats" in h:
                vs = h["value_stats"]
                value_stats[i] = torch.tensor([
                    float(vs.get("mean_norm", 0.0)),
                    float(vs.get("std_norm", 0.0)),
                    float(vs.get("min_norm", 0.0)),
                    float(vs.get("max_norm", 0.0)),
                    float(vs.get("null_ratio", 0.0)),
                    float(vs.get("is_numeric", 0.0)),
                ], dtype=torch.float)
        return header_ids, value_stats

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        workspace = sample["workspace"][:self.max_workspace_size]
        relevant_set = set(sample["relevant_subset"])

        q_enc = self.tokenizer(
            sample["query"],
            max_length=self.max_query_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        ws_header_ids = torch.zeros(
            self.max_workspace_size, self.max_cols, self.max_col_length, dtype=torch.long
        )
        ws_value_stats = torch.zeros(
            self.max_workspace_size, self.max_cols, self.value_dim, dtype=torch.float
        )
        node_mask = torch.zeros(self.max_workspace_size, dtype=torch.float)
        labels = torch.zeros(self.max_workspace_size, dtype=torch.float)

        for i, sid in enumerate(workspace):
            ids, stats = self._encode_sheet(sid)
            ws_header_ids[i] = ids
            ws_value_stats[i] = stats
            node_mask[i] = 1.0
            labels[i] = 1.0 if sid in relevant_set else 0.0

        return {
            "query_ids": q_enc["input_ids"][0],
            "query_mask": q_enc["attention_mask"][0],
            "workspace_header_ids": ws_header_ids,
            "workspace_value_stats": ws_value_stats,
            "node_mask": node_mask,
            "labels": labels,
        }


# =============================================================================
# 损失函数
# =============================================================================

class InfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, emb_a: torch.Tensor, emb_b: torch.Tensor) -> torch.Tensor:
        batch_size = emb_a.shape[0]
        a = F.normalize(emb_a, dim=-1)
        b = F.normalize(emb_b, dim=-1)
        logits = torch.matmul(a, b.T) / self.temperature
        labels = torch.arange(batch_size, device=emb_a.device)
        loss_ab = F.cross_entropy(logits, labels)
        loss_ba = F.cross_entropy(logits.T, labels)
        return 0.5 * (loss_ab + loss_ba)


# =============================================================================
# 工具函数
# =============================================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def path_exists(path_value) -> bool:
    return bool(path_value) and os.path.exists(path_value)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_features_map(data_cfg: Dict) -> Dict[str, Dict]:
    features_files = []
    if isinstance(data_cfg.get("features_files"), list):
        features_files.extend(data_cfg["features_files"])
    if data_cfg.get("features_file"):
        features_files.append(data_cfg["features_file"])

    features_files = [p for p in features_files if p]
    if not features_files:
        raise ValueError("data.features_file 或 data.features_files 至少需要提供一个特征文件路径。")

    merged = {}
    file_counts = []
    for path in features_files:
        if not os.path.exists(path):
            raise FileNotFoundError(f"未找到特征文件: {path}")
        items = load_json(path)
        for item in items:
            sheet_id = item["sheet_id"]
            merged[sheet_id] = item
        file_counts.append((path, len(items)))

    print("✓ 已加载特征文件:")
    for path, count in file_counts:
        print(f"  - {path}: {count} 条")
    print(f"✓ 合并后唯一 sheet 特征数: {len(merged)}")
    return merged


def infer_num_workers(cfg: Dict) -> int:
    data_cfg = cfg.get("data", {})
    if "num_workers" in data_cfg:
        return int(data_cfg["num_workers"])
    return 0 if not torch.cuda.is_available() else min(4, os.cpu_count() or 1)


def print_missing_feature_report(dataset, name: str):
    if getattr(dataset, "missing_sheet_ids", None):
        missing = sorted(dataset.missing_sheet_ids)
        preview = ", ".join(missing[:10])
        print(f"⚠ {name} 有 {len(missing)} 个 sheet_id 未在 features 中找到。前几个: {preview}")
    else:
        print(f"✓ {name} 所有 sheet_id 都能在 features 中找到")


def get_amp_dtype(device: torch.device):
    if device.type != "cuda":
        return None
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def make_autocast_context(device: torch.device, enabled: bool):
    if not enabled or device.type != "cuda":
        return autocast(device_type="cpu", enabled=False)
    return autocast(device_type="cuda", dtype=get_amp_dtype(device), enabled=True)


# =============================================================================
# 训练器
# =============================================================================

class AgentSheetTrainer:
    def __init__(self, cfg: Dict, model: AgentSheetModel, tokenizer, device: torch.device):
        self.cfg = cfg
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.use_wandb = cfg.get("wandb", {}).get("enabled", False)
        self.wandb = None

        if self.use_wandb:
            try:
                import wandb
                wandb.init(
                    project=cfg["wandb"].get("project", "agentsheet"),
                    name=cfg["experiment"]["name"],
                    config=cfg,
                    tags=cfg["wandb"].get("tags", []),
                )
                self.wandb = wandb
            except Exception as exc:
                print(f"⚠ wandb 初始化失败，已跳过: {exc}")
                self.use_wandb = False

    def _log(self, metrics: Dict, step: int):
        if self.use_wandb and self.wandb is not None:
            self.wandb.log(metrics, step=step)

    def _save_checkpoint(self, path: str, extra: Dict = None):
        ensure_dir(os.path.dirname(path))
        state = {"model": self.model.state_dict()}
        if extra:
            state.update(extra)
        torch.save(state, path)
        print(f"✓ Checkpoint 已保存: {path}")

    def train_stage1(self, train_loader: DataLoader, eval_loader: DataLoader):
        s1_cfg = self.cfg["stage1"]
        output_dir = os.path.join(
            self.cfg["experiment"]["output_dir"],
            self.cfg["experiment"]["name"],
            "stage1",
        )
        ensure_dir(output_dir)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(s1_cfg["lr"]),
            weight_decay=float(s1_cfg["weight_decay"]),
        )
        total_steps = max(1, len(train_loader) * int(s1_cfg["epochs"]))
        warmup_steps = int(total_steps * float(s1_cfg.get("warmup_ratio", 0.1)))
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

        infonce = InfoNCELoss(temperature=float(s1_cfg.get("temperature", 0.07)))
        lambda_align = float(s1_cfg.get("lambda_align", 0.5))
        use_amp = bool(s1_cfg.get("fp16", False)) and self.device.type == "cuda"
        scaler = GradScaler(device="cuda", enabled=use_amp)

        best_metric = -1.0
        best_path = os.path.join(output_dir, "best.pt")
        final_path = os.path.join(output_dir, "final.pt")
        global_step = 0

        print(f"\n{'=' * 60}")
        print(f"Stage 1 预训练: {s1_cfg['epochs']} epochs, {total_steps} steps")
        print(f"AMP enabled: {use_amp}")
        print(f"{'=' * 60}")

        for epoch in range(int(s1_cfg["epochs"])):
            self.model.train()
            epoch_loss = 0.0
            pbar = tqdm(train_loader, desc=f"Stage1 Epoch {epoch + 1}/{s1_cfg['epochs']}")

            for batch in pbar:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                optimizer.zero_grad(set_to_none=True)

                with make_autocast_context(self.device, use_amp):
                    out = self.model(
                        mode="pairwise",
                        header_ids_a=batch["header_ids_a"],
                        value_stats_a=batch["value_stats_a"],
                        header_ids_b=batch["header_ids_b"],
                        value_stats_b=batch["value_stats_b"],
                    )
                    pos_mask = batch["label"] > 0.5
                    if int(pos_mask.sum().item()) > 1:
                        loss_infonce = infonce(out["emb_a"][pos_mask], out["emb_b"][pos_mask])
                    else:
                        loss_infonce = torch.zeros((), device=self.device)

                    loss_align = F.binary_cross_entropy_with_logits(out["sim_score"], batch["label"])
                    loss = loss_infonce + lambda_align * loss_align

                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(s1_cfg["max_grad_norm"]))
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(s1_cfg["max_grad_norm"]))
                    optimizer.step()
                scheduler.step()

                epoch_loss += float(loss.item())
                global_step += 1
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

                if global_step % int(self.cfg.get("wandb", {}).get("log_interval", 50)) == 0:
                    self._log(
                        {
                            "stage1/loss": float(loss.item()),
                            "stage1/loss_infonce": float(loss_infonce.item()),
                            "stage1/loss_align": float(loss_align.item()),
                            "stage1/lr": float(scheduler.get_last_lr()[0]),
                        },
                        global_step,
                    )

                if len(eval_loader) > 0 and global_step % int(s1_cfg.get("eval_steps", 500)) == 0:
                    metrics = self._eval_stage1(eval_loader, use_amp)
                    print(f"\n[Stage1][step={global_step}] eval: {metrics}")
                    self._log({f"stage1/eval/{k}": v for k, v in metrics.items()}, global_step)
                    primary = float(metrics.get("align_auc", 0.0))
                    if primary >= best_metric:
                        best_metric = primary
                        self._save_checkpoint(best_path, {"epoch": epoch, "step": global_step, "metric": best_metric})

            avg_loss = epoch_loss / max(1, len(train_loader))
            print(f"Stage1 Epoch {epoch + 1} 完成，平均 Loss: {avg_loss:.4f}")

            if len(eval_loader) > 0:
                metrics = self._eval_stage1(eval_loader, use_amp)
                print(f"[Stage1][epoch_end={epoch + 1}] eval: {metrics}")
                primary = float(metrics.get("align_auc", 0.0))
                if primary >= best_metric:
                    best_metric = primary
                    self._save_checkpoint(best_path, {"epoch": epoch, "step": global_step, "metric": best_metric})
            elif not os.path.exists(best_path):
                self._save_checkpoint(best_path, {"epoch": epoch, "step": global_step, "metric": avg_loss})

        self._save_checkpoint(final_path, {"epoch": int(s1_cfg["epochs"]), "step": global_step})
        if not os.path.exists(best_path):
            self._save_checkpoint(best_path, {"epoch": int(s1_cfg["epochs"]), "step": global_step, "metric": 0.0})

        print(f"\n✓ Stage 1 训练完成，最佳指标: {best_metric:.4f}")
        return best_path

    def _eval_stage1(self, eval_loader: DataLoader, use_amp: bool) -> Dict[str, float]:
        self.model.eval()
        all_scores = []
        all_labels = []

        with torch.no_grad():
            for batch in eval_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                with make_autocast_context(self.device, use_amp):
                    out = self.model(
                        mode="pairwise",
                        header_ids_a=batch["header_ids_a"],
                        value_stats_a=batch["value_stats_a"],
                        header_ids_b=batch["header_ids_b"],
                        value_stats_b=batch["value_stats_b"],
                    )
                all_scores.extend(torch.sigmoid(out["sim_score"]).detach().cpu().tolist())
                all_labels.extend(batch["label"].detach().cpu().tolist())

        self.model.train()
        if len(set(float(x) for x in all_labels)) < 2:
            return {"align_auc": 0.0}
        try:
            from sklearn.metrics import roc_auc_score
            auc = float(roc_auc_score(all_labels, all_scores))
        except Exception:
            auc = 0.0
        return {"align_auc": auc}

    def train_stage2(self, train_loader: DataLoader, eval_loader: DataLoader):
        s2_cfg = self.cfg["stage2"]
        output_dir = os.path.join(
            self.cfg["experiment"]["output_dir"],
            self.cfg["experiment"]["name"],
            "stage2",
        )
        ensure_dir(output_dir)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(s2_cfg["lr"]),
            weight_decay=float(s2_cfg["weight_decay"]),
        )
        total_steps = max(1, len(train_loader) * int(s2_cfg["epochs"]))
        warmup_steps = int(total_steps * float(s2_cfg.get("warmup_ratio", 0.1)))
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

        use_amp = bool(s2_cfg.get("fp16", False)) and self.device.type == "cuda"
        scaler = GradScaler(device="cuda", enabled=use_amp)
        pos_weight = torch.tensor(float(s2_cfg.get("pos_weight", 3.0)), device=self.device)
        best_metric = -1.0
        global_step = 0
        best_path = os.path.join(output_dir, "best.pt")
        final_path = os.path.join(output_dir, "final.pt")
        primary_metric_name = self.cfg.get("evaluation", {}).get("primary_metric", "Recall@3")

        print(f"\n{'=' * 60}")
        print(f"Stage 2 微调: {s2_cfg['epochs']} epochs, {total_steps} steps")
        print(f"AMP enabled: {use_amp}")
        print(f"{'=' * 60}")

        for epoch in range(int(s2_cfg["epochs"])):
            self.model.train()
            epoch_loss = 0.0
            pbar = tqdm(train_loader, desc=f"Stage2 Epoch {epoch + 1}/{s2_cfg['epochs']}")

            for batch in pbar:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                optimizer.zero_grad(set_to_none=True)

                with make_autocast_context(self.device, use_amp):
                    out = self.model(
                        mode="nway",
                        query_ids=batch["query_ids"],
                        query_mask=batch["query_mask"],
                        workspace_header_ids=batch["workspace_header_ids"],
                        workspace_value_stats=batch["workspace_value_stats"],
                        node_mask=batch["node_mask"],
                    )
                    loss_raw = F.binary_cross_entropy_with_logits(
                        out["activation_logits"],
                        batch["labels"],
                        pos_weight=pos_weight,
                        reduction="none",
                    )
                    loss = (loss_raw * batch["node_mask"]).sum() / batch["node_mask"].sum().clamp(min=1.0)

                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(s2_cfg["max_grad_norm"]))
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(s2_cfg["max_grad_norm"]))
                    optimizer.step()
                scheduler.step()

                epoch_loss += float(loss.item())
                global_step += 1
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

                if global_step % int(self.cfg.get("wandb", {}).get("log_interval", 50)) == 0:
                    self._log(
                        {
                            "stage2/loss": float(loss.item()),
                            "stage2/lr": float(scheduler.get_last_lr()[0]),
                        },
                        global_step,
                    )

                if len(eval_loader) > 0 and global_step % int(s2_cfg.get("eval_steps", 200)) == 0:
                    metrics = self._eval_stage2(eval_loader, use_amp, float(s2_cfg.get("activation_threshold", 0.5)))
                    print(f"\n[Stage2][step={global_step}] eval: {metrics}")
                    self._log({f"stage2/eval/{k}": v for k, v in metrics.items()}, global_step)
                    primary = float(metrics.get(primary_metric_name, 0.0))
                    if primary >= best_metric:
                        best_metric = primary
                        self._save_checkpoint(best_path, {"epoch": epoch, "step": global_step, "metric": best_metric})

            avg_loss = epoch_loss / max(1, len(train_loader))
            print(f"Stage2 Epoch {epoch + 1} 完成，平均 Loss: {avg_loss:.4f}")

            if len(eval_loader) > 0:
                metrics = self._eval_stage2(eval_loader, use_amp, float(s2_cfg.get("activation_threshold", 0.5)))
                print(f"[Stage2][epoch_end={epoch + 1}] eval: {metrics}")
                primary = float(metrics.get(primary_metric_name, 0.0))
                if primary >= best_metric:
                    best_metric = primary
                    self._save_checkpoint(best_path, {"epoch": epoch, "step": global_step, "metric": best_metric})
            elif not os.path.exists(best_path):
                self._save_checkpoint(best_path, {"epoch": epoch, "step": global_step, "metric": avg_loss})

        self._save_checkpoint(final_path, {"epoch": int(s2_cfg["epochs"]), "step": global_step})
        if not os.path.exists(best_path):
            self._save_checkpoint(best_path, {"epoch": int(s2_cfg["epochs"]), "step": global_step, "metric": 0.0})

        print(f"\n✓ Stage 2 训练完成，最佳指标: {best_metric:.4f}")
        return best_path

    def _eval_stage2(self, eval_loader: DataLoader, use_amp: bool, threshold: float) -> Dict[str, float]:
        self.model.eval()
        all_retrieved = []
        all_predicted = []
        all_relevant = []

        with torch.no_grad():
            for batch in eval_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                with make_autocast_context(self.device, use_amp):
                    out = self.model(
                        mode="nway",
                        query_ids=batch["query_ids"],
                        query_mask=batch["query_mask"],
                        workspace_header_ids=batch["workspace_header_ids"],
                        workspace_value_stats=batch["workspace_value_stats"],
                        node_mask=batch["node_mask"],
                    )

                logits = out["activation_logits"]
                node_mask = batch["node_mask"]
                labels = batch["labels"]

                for b in range(logits.shape[0]):
                    valid_n = int(node_mask[b].sum().item())
                    if valid_n <= 0:
                        continue
                    scores = torch.sigmoid(logits[b, :valid_n]).detach().cpu().tolist()
                    true_labels = labels[b, :valid_n].detach().cpu().tolist()

                    ranked = sorted(range(valid_n), key=lambda i: scores[i], reverse=True)
                    relevant = [i for i, x in enumerate(true_labels) if x > 0.5]
                    predicted = [i for i in range(valid_n) if scores[i] > threshold]
                    if not predicted and ranked:
                        predicted = ranked[:1]

                    all_retrieved.append(ranked)
                    all_predicted.append(predicted)
                    all_relevant.append(relevant)

        self.model.train()
        if not all_retrieved:
            return {"Recall@1": 0.0, "Recall@3": 0.0, "Recall@5": 0.0, "MRR": 0.0, "NDCG@1": 0.0, "NDCG@3": 0.0, "NDCG@5": 0.0, "P@1": 0.0, "P@3": 0.0, "P@5": 0.0, "Subset_EM": 0.0, "Subset_F1": 0.0, "Jaccard": 0.0}

        retrieval = RetrievalMetrics.compute_all(all_retrieved, all_relevant, ks=[1, 3, 5])
        subset = SubsetMetrics.compute_all(all_predicted, all_relevant)
        return {**retrieval, **subset}


# =============================================================================
# 主入口
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="AgentSheet 训练脚本")
    parser.add_argument("--config", required=True, help="YAML 配置文件路径")
    parser.add_argument("--stage", type=int, choices=[1, 2], default=None, help="只运行指定阶段")
    parser.add_argument("--resume", default=None, help="从 checkpoint 恢复")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.resume:
        cfg.setdefault("experiment", {})["resume"] = args.resume

    set_seed(int(cfg["experiment"].get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✓ 设备: {device}")
    if device.type == "cuda":
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")

    data_cfg = cfg["data"]
    print("正在加载数据...")
    features_map = load_features_map(data_cfg)

    model_cfg = AgentSheetConfig(**cfg["model"])
    model = AgentSheetModel(model_cfg).to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model"]["backbone_name"],
        local_files_only=cfg["model"].get("local_files_only", False),
    )

    if cfg.get("experiment", {}).get("resume"):
        state = torch.load(cfg["experiment"]["resume"], map_location=device)
        model.load_state_dict(state["model"], strict=False)
        print(f"✓ 从 checkpoint 恢复: {cfg['experiment']['resume']}")

    trainer = AgentSheetTrainer(cfg, model, tokenizer, device)
    num_workers = infer_num_workers(cfg)
    pin_memory = device.type == "cuda"

    stage1_ckpt = None

    if (args.stage is None or args.stage == 1) and cfg.get("stage1", {}).get("enabled", True):
        pairwise_train_path = data_cfg.get("pairwise_train")
        pairwise_eval_path = data_cfg.get("pairwise_eval")
        if not path_exists(pairwise_train_path):
            raise FileNotFoundError(f"未找到 Stage 1 训练文件: {pairwise_train_path}")
        if not path_exists(pairwise_eval_path):
            raise FileNotFoundError(f"未找到 Stage 1 评估文件: {pairwise_eval_path}")

        train_pairs = load_json(pairwise_train_path)
        eval_pairs = load_json(pairwise_eval_path)

        train_ds = PairwiseDataset(
            train_pairs,
            features_map,
            tokenizer,
            max_cols=int(data_cfg.get("max_cols", 16)),
            max_col_length=int(data_cfg.get("max_col_length", 32)),
            value_dim=int(cfg["model"].get("value_dim", 6)),
        )
        eval_ds = PairwiseDataset(
            eval_pairs,
            features_map,
            tokenizer,
            max_cols=int(data_cfg.get("max_cols", 16)),
            max_col_length=int(data_cfg.get("max_col_length", 32)),
            value_dim=int(cfg["model"].get("value_dim", 6)),
        )

        # 预扫描一次，方便尽早暴露 feature 漏洞
        _ = train_ds[0] if len(train_ds) > 0 else None
        _ = eval_ds[0] if len(eval_ds) > 0 else None
        print_missing_feature_report(train_ds, "Stage 1 train")
        print_missing_feature_report(eval_ds, "Stage 1 eval")

        train_loader = DataLoader(
            train_ds,
            batch_size=int(cfg["stage1"].get("batch_size", 32)),
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        eval_loader = DataLoader(
            eval_ds,
            batch_size=max(1, int(cfg["stage1"].get("batch_size", 32)) * 2),
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        print(f"✓ Stage 1 训练集: {len(train_ds)} 条，评估集: {len(eval_ds)} 条")
        stage1_ckpt = trainer.train_stage1(train_loader, eval_loader)

        if args.stage is None and path_exists(stage1_ckpt):
            state = torch.load(stage1_ckpt, map_location=device)
            model.load_state_dict(state["model"], strict=False)
            print("✓ Stage 2 将从 Stage 1 最佳模型初始化")

    should_try_stage2 = (args.stage is None or args.stage == 2) and cfg.get("stage2", {}).get("enabled", True)
    if should_try_stage2:
        nway_train_path = data_cfg.get("nway_train")
        nway_eval_path = data_cfg.get("nway_eval")
        if not path_exists(nway_train_path) or not path_exists(nway_eval_path):
            print("⚠ Stage 2 所需的 nway_train / nway_eval 文件不存在，已自动跳过 Stage 2。")
        else:
            nway_train = load_json(nway_train_path)
            nway_eval = load_json(nway_eval_path)

            train_ds = NwayDataset(
                nway_train,
                features_map,
                tokenizer,
                max_workspace_size=int(data_cfg.get("max_workspace_size", cfg["model"].get("max_workspace_size", 10))),
                max_cols=int(data_cfg.get("max_cols", 16)),
                max_col_length=int(data_cfg.get("max_col_length", 32)),
                max_query_length=int(data_cfg.get("max_query_length", 64)),
                value_dim=int(cfg["model"].get("value_dim", 6)),
            )
            eval_ds = NwayDataset(
                nway_eval,
                features_map,
                tokenizer,
                max_workspace_size=int(data_cfg.get("max_workspace_size", cfg["model"].get("max_workspace_size", 10))),
                max_cols=int(data_cfg.get("max_cols", 16)),
                max_col_length=int(data_cfg.get("max_col_length", 32)),
                max_query_length=int(data_cfg.get("max_query_length", 64)),
                value_dim=int(cfg["model"].get("value_dim", 6)),
            )

            _ = train_ds[0] if len(train_ds) > 0 else None
            _ = eval_ds[0] if len(eval_ds) > 0 else None
            print_missing_feature_report(train_ds, "Stage 2 train")
            print_missing_feature_report(eval_ds, "Stage 2 eval")

            train_loader = DataLoader(
                train_ds,
                batch_size=int(cfg["stage2"].get("batch_size", 8)),
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
            eval_loader = DataLoader(
                eval_ds,
                batch_size=max(1, int(cfg["stage2"].get("batch_size", 8)) * 2),
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )

            print(f"✓ Stage 2 训练集: {len(train_ds)} 条，评估集: {len(eval_ds)} 条")
            trainer.train_stage2(train_loader, eval_loader)

    print("\n✓ 训练流程结束")
    print(f"  输出目录: {cfg['experiment']['output_dir']}/{cfg['experiment']['name']}/")


if __name__ == "__main__":
    main()
