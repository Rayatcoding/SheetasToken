#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage2 GTN-lite / Full-GTN training with DDP support.

Recommended placement:
repo_root/stage2_gtn.py

Expected data files under --data-dir:
- nway_train.json
- nway_eval.json
- sheet_features_train.json / sheet_features_eval.json / sheet_features.json

Works with the Stage1 biencoder_model.py replacement. It reuses the same HF backbone
family and can optionally load Stage1 weights from best_model/final_model/classifier.pt.
"""

import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def setup_distributed() -> torch.device:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if not dist.is_initialized():
            dist.init_process_group(backend=backend)
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            return torch.device("cuda", local_rank)
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cleanup_distributed() -> None:
    if is_dist():
        dist.barrier()
        dist.destroy_process_group()


def reduce_mean(value: float, device: torch.device) -> float:
    if not is_dist():
        return float(value)
    t = torch.tensor(float(value), device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= get_world_size()
    return t.item()


@dataclass
class EvalMetrics:
    loss: float
    recall_at_1: float
    recall_at_3: float
    f1: float
    subset_acc: float


class SheetTextSerializer:
    def __init__(
        self,
        sheet_feature_map: Dict[str, Dict],
        max_header_texts: int = 12,
        include_shape_feature: bool = True,
        include_source_feature: bool = True,
    ) -> None:
        self.sheet_feature_map = sheet_feature_map
        self.max_header_texts = max_header_texts
        self.include_shape_feature = include_shape_feature
        self.include_source_feature = include_source_feature

    def to_text(self, sheet_id: str) -> str:
        feat = self.sheet_feature_map.get(sheet_id)
        if not feat:
            return str(sheet_id)
        segments: List[str] = []
        if self.include_source_feature and feat.get("source"):
            segments.append(f"source: {feat['source']}")
        if self.include_shape_feature:
            nr = feat.get("num_rows", "?")
            nc = feat.get("num_cols", "?")
            segments.append(f"shape: {nr}x{nc}")
        headers = feat.get("headers", [])
        header_texts: List[str] = []
        for h in headers[: self.max_header_texts]:
            txt = str(h.get("text", "")).strip() if isinstance(h, dict) else str(h).strip()
            if txt:
                header_texts.append(txt)
        if header_texts:
            segments.append("headers: " + " | ".join(header_texts))
        return " ; ".join(segments) if segments else str(sheet_id)

    def header_set(self, sheet_id: str) -> set:
        feat = self.sheet_feature_map.get(sheet_id)
        if not feat:
            return set()
        headers = feat.get("headers", [])
        out = set()
        for h in headers[: self.max_header_texts]:
            txt = str(h.get("text", "")).strip().lower() if isinstance(h, dict) else str(h).strip().lower()
            if txt:
                out.add(txt)
        return out

    def source(self, sheet_id: str) -> str:
        feat = self.sheet_feature_map.get(sheet_id, {})
        return str(feat.get("source", ""))

    def shape(self, sheet_id: str) -> Tuple[float, float]:
        feat = self.sheet_feature_map.get(sheet_id, {})
        return float(feat.get("num_rows", 0.0) or 0.0), float(feat.get("num_cols", 0.0) or 0.0)


class NWayDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        split: str,
        tokenizer,
        max_length: int = 256,
        max_query_length: int = 64,
        max_workspace_size: int = 10,
        features_file: Optional[str] = None,
        max_header_texts: int = 12,
        include_shape_feature: bool = True,
        include_source_feature: bool = True,
    ) -> None:
        self.data_dir = data_dir
        self.split = split
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_query_length = max_query_length
        self.max_workspace_size = max_workspace_size
        self.features_file = features_file
        self.serializer = SheetTextSerializer(
            self._load_sheet_feature_map(),
            max_header_texts=max_header_texts,
            include_shape_feature=include_shape_feature,
            include_source_feature=include_source_feature,
        )
        self.data = self._load_data()

    def _split_to_filename(self) -> str:
        return os.path.join(self.data_dir, f"{self.split}.json")

    def _load_data(self) -> List[Dict]:
        path = self._split_to_filename()
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing dataset file: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"{path} must contain a JSON list.")
        return data

    def _infer_features_file(self) -> Optional[str]:
        if self.features_file:
            return self.features_file
        split_name = self.split
        if split_name.startswith("nway_"):
            split_name = split_name[len("nway_"):]
        candidate = os.path.join(self.data_dir, f"sheet_features_{split_name}.json")
        if os.path.exists(candidate):
            return candidate
        fallback = os.path.join(self.data_dir, "sheet_features.json")
        if os.path.exists(fallback):
            return fallback
        return None

    def _load_sheet_feature_map(self) -> Dict[str, Dict]:
        path = self._infer_features_file()
        if not path:
            return {}
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        out = {}
        for item in raw:
            sid = item.get("sheet_id")
            if sid:
                out[sid] = item
        return out

    def __len__(self) -> int:
        return len(self.data)

    def _encode_text(self, text: str, max_length: int) -> Dict[str, torch.Tensor]:
        enc = self.tokenizer(
            text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        out = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }
        if "token_type_ids" in enc:
            out["token_type_ids"] = enc["token_type_ids"].squeeze(0)
        return out

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        query_text = item["query"]
        workspace = item["workspace"][: self.max_workspace_size]
        relevant = set(item.get("relevant_subset", []))

        q = self._encode_text(query_text, self.max_query_length)

        ws_input_ids = []
        ws_attention_mask = []
        ws_token_type_ids = []
        node_mask = []
        labels = []
        sheet_ids = []

        for i in range(self.max_workspace_size):
            if i < len(workspace):
                sid = workspace[i]
                text = self.serializer.to_text(sid)
                enc = self._encode_text(text, self.max_length)
                ws_input_ids.append(enc["input_ids"])
                ws_attention_mask.append(enc["attention_mask"])
                ws_token_type_ids.append(enc.get("token_type_ids", torch.zeros_like(enc["input_ids"])))
                node_mask.append(1.0)
                labels.append(1.0 if sid in relevant else 0.0)
                sheet_ids.append(sid)
            else:
                ws_input_ids.append(torch.zeros(self.max_length, dtype=torch.long))
                ws_attention_mask.append(torch.zeros(self.max_length, dtype=torch.long))
                ws_token_type_ids.append(torch.zeros(self.max_length, dtype=torch.long))
                node_mask.append(0.0)
                labels.append(0.0)
                sheet_ids.append("")

        return {
            "query_input_ids": q["input_ids"],
            "query_attention_mask": q["attention_mask"],
            "query_token_type_ids": q.get("token_type_ids", torch.zeros_like(q["input_ids"])),
            "workspace_input_ids": torch.stack(ws_input_ids, dim=0),
            "workspace_attention_mask": torch.stack(ws_attention_mask, dim=0),
            "workspace_token_type_ids": torch.stack(ws_token_type_ids, dim=0),
            "node_mask": torch.tensor(node_mask, dtype=torch.float),
            "labels": torch.tensor(labels, dtype=torch.float),
            "sheet_ids": sheet_ids,
        }


class TextBiEncoder(nn.Module):
    def __init__(
        self,
        model_name: str,
        local_files_only: bool = True,
        embedding_strategy: str = "cls",
        use_layer_mix: bool = False,
        use_extra_position_embedding: bool = False,
        position_embedding_scale: float = 1.0,
        max_length: int = 256,
        normalize_embeddings: bool = True,
    ) -> None:
        super().__init__()
        self.embedding_strategy = embedding_strategy
        self.use_layer_mix = use_layer_mix
        self.use_extra_position_embedding = use_extra_position_embedding
        self.position_embedding_scale = position_embedding_scale
        self.normalize_embeddings = normalize_embeddings

        self.backbone = AutoModel.from_pretrained(model_name, local_files_only=local_files_only)
        hidden_size = self.backbone.config.hidden_size
        self.hidden_size = hidden_size
        self.embedding_dim = self._get_embedding_dim(hidden_size)

        if self.use_extra_position_embedding:
            self.extra_position_embedding = nn.Embedding(max_length, hidden_size)
        if self.use_layer_mix:
            num_layers = self.backbone.config.num_hidden_layers + 1
            self.layer_weights = nn.Parameter(torch.zeros(num_layers))

    def _get_embedding_dim(self, hidden_size: int) -> int:
        if self.embedding_strategy in {"cls", "mean", "max"}:
            return hidden_size
        if self.embedding_strategy in {"cls_mean_concat", "mean_max_concat"}:
            return hidden_size * 2
        if self.embedding_strategy == "cls_mean_max_concat":
            return hidden_size * 3
        raise ValueError(f"Unsupported embedding_strategy: {self.embedding_strategy}")

    def _masked_mean_pool(self, sequence_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).float()
        summed = (sequence_output * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def _masked_max_pool(self, sequence_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).bool()
        masked = sequence_output.masked_fill(~mask, -1e9)
        return masked.max(dim=1).values

    def _build_embedding(self, sequence_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        cls_emb = sequence_output[:, 0, :]
        mean_emb = self._masked_mean_pool(sequence_output, attention_mask)
        max_emb = self._masked_max_pool(sequence_output, attention_mask)

        if self.embedding_strategy == "cls":
            emb = cls_emb
        elif self.embedding_strategy == "mean":
            emb = mean_emb
        elif self.embedding_strategy == "max":
            emb = max_emb
        elif self.embedding_strategy == "cls_mean_concat":
            emb = torch.cat([cls_emb, mean_emb], dim=-1)
        elif self.embedding_strategy == "mean_max_concat":
            emb = torch.cat([mean_emb, max_emb], dim=-1)
        elif self.embedding_strategy == "cls_mean_max_concat":
            emb = torch.cat([cls_emb, mean_emb, max_emb], dim=-1)
        else:
            raise ValueError(f"Unsupported embedding_strategy: {self.embedding_strategy}")

        if self.normalize_embeddings:
            emb = F.normalize(emb, dim=-1)
        return emb

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=self.use_layer_mix,
            return_dict=True,
        )
        if self.use_layer_mix:
            hidden_states = torch.stack(outputs.hidden_states, dim=0)
            layer_probs = torch.softmax(self.layer_weights, dim=0).view(-1, 1, 1, 1)
            sequence_output = (hidden_states * layer_probs).sum(dim=0)
        else:
            sequence_output = outputs.last_hidden_state
        if self.use_extra_position_embedding:
            seq_len = sequence_output.size(1)
            position_ids = torch.arange(seq_len, device=sequence_output.device).unsqueeze(0)
            pos_emb = self.extra_position_embedding(position_ids)
            sequence_output = sequence_output + self.position_embedding_scale * pos_emb
        return self._build_embedding(sequence_output, attention_mask)


class DenseGATLayer(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1, negative_slope: float = 0.2) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=False)
        self.attn_src = nn.Linear(dim, 1, bias=False)
        self.attn_dst = nn.Linear(dim, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor, node_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.proj(x)
        src_logits = self.attn_src(h)
        dst_logits = self.attn_dst(h)
        e = src_logits + dst_logits.transpose(1, 2)
        e = self.leaky_relu(e)

        valid_pair = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
        no_edge = (adj <= 0) | (valid_pair <= 0)
        e = e.masked_fill(no_edge, -1e9)
        alpha = torch.softmax(e, dim=-1)
        alpha = alpha * valid_pair
        alpha = alpha * (adj > 0).float()
        alpha = alpha / alpha.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        alpha = self.dropout(alpha)

        out = torch.bmm(alpha, h)
        out = self.out_proj(out)
        out = self.norm(out + x)
        out = out * node_mask.unsqueeze(-1)
        return out, alpha


class AdjacencyBuilder(nn.Module):
    def __init__(self, mode: str = "gtn_lite", dropout: float = 0.1) -> None:
        super().__init__()
        self.mode = mode
        self.dropout = nn.Dropout(dropout)
        self.num_channels = 4
        if mode == "gtn_lite":
            self.channel_logits = nn.Parameter(torch.zeros(self.num_channels))
        elif mode == "full_gtn":
            self.channel_logits_q1 = nn.Parameter(torch.zeros(self.num_channels))
            self.channel_logits_q2 = nn.Parameter(torch.zeros(self.num_channels))
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    @staticmethod
    def _row_normalize(adj: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        valid_pair = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
        adj = adj * valid_pair
        eye = torch.eye(adj.size(-1), device=adj.device).unsqueeze(0)
        adj = adj * (1 - eye)
        row_sum = adj.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        return adj / row_sum

    def forward(self, channels: torch.Tensor, node_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        # channels: (B, K, N, N)
        if self.mode == "gtn_lite":
            w = torch.softmax(self.channel_logits, dim=0)
            fused = (channels * w.view(1, -1, 1, 1)).sum(dim=1)
            fused = F.relu(fused)
            fused = self._row_normalize(fused, node_mask)
            aux = {"channel_weights": w.detach()}
            return fused, fused, aux

        w1 = torch.softmax(self.channel_logits_q1, dim=0)
        w2 = torch.softmax(self.channel_logits_q2, dim=0)
        q1 = (channels * w1.view(1, -1, 1, 1)).sum(dim=1)
        q2 = (channels * w2.view(1, -1, 1, 1)).sum(dim=1)
        meta = torch.bmm(F.relu(q1), F.relu(q2))
        meta = self._row_normalize(meta, node_mask)
        prior = self._row_normalize(F.relu(q1), node_mask)
        aux = {"channel_weights_q1": w1.detach(), "channel_weights_q2": w2.detach()}
        return meta, prior, aux


class Stage2GTNModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        graph_mode: str = "gtn_lite",
        local_files_only: bool = True,
        embedding_strategy: str = "cls",
        use_layer_mix: bool = False,
        use_extra_position_embedding: bool = False,
        position_embedding_scale: float = 1.0,
        max_length: int = 256,
        normalize_embeddings: bool = True,
        graph_dropout: float = 0.1,
        num_gat_layers: int = 1,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = TextBiEncoder(
            model_name=model_name,
            local_files_only=local_files_only,
            embedding_strategy=embedding_strategy,
            use_layer_mix=use_layer_mix,
            use_extra_position_embedding=use_extra_position_embedding,
            position_embedding_scale=position_embedding_scale,
            max_length=max_length,
            normalize_embeddings=normalize_embeddings,
        )
        dim = self.encoder.embedding_dim
        self.query_proj = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.LayerNorm(dim))
        self.node_proj = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.LayerNorm(dim))
        self.adj_builder = AdjacencyBuilder(mode=graph_mode, dropout=graph_dropout)
        self.gat_layers = nn.ModuleList([DenseGATLayer(dim, dropout=graph_dropout) for _ in range(num_gat_layers)])
        self.scorer = nn.Sequential(
            nn.Linear(dim * 4, dim),
            nn.GELU(),
            nn.Dropout(graph_dropout),
            nn.Linear(dim, 1),
        )
        self.graph_mode = graph_mode

        if freeze_backbone:
            for p in self.encoder.backbone.parameters():
                p.requires_grad = False

    def load_stage1_checkpoint(self, ckpt_path: str) -> None:
        if not ckpt_path or not os.path.exists(ckpt_path):
            return
        state = torch.load(ckpt_path, map_location="cpu")
        state_dict = state.get("state_dict", state)
        encoder_state = self.encoder.state_dict()
        mapped = {}
        for k, v in state_dict.items():
            if k.startswith("backbone."):
                new_k = f"encoder.{k}"
            else:
                continue
            if new_k in encoder_state and encoder_state[new_k].shape == v.shape:
                mapped[new_k] = v
        missing, unexpected = self.load_state_dict(mapped, strict=False)
        if is_main_process():
            print(f"Loaded Stage1 backbone weights from {ckpt_path}; matched={len(mapped)} missing={len(missing)} unexpected={len(unexpected)}", flush=True)

    def _build_channels(
        self,
        query_emb: torch.Tensor,
        node_embs: torch.Tensor,
        node_mask: torch.Tensor,
        schema_prior: torch.Tensor,
        source_prior: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        norm_nodes = F.normalize(node_embs, dim=-1)
        q = F.normalize(query_emb, dim=-1)

        sem = torch.bmm(norm_nodes, norm_nodes.transpose(1, 2))
        sem = (sem + 1.0) * 0.5

        q_sim = (norm_nodes * q.unsqueeze(1)).sum(dim=-1)
        q_graph = torch.einsum("bi,bj->bij", q_sim, q_sim)
        q_graph = (q_graph + 1.0) * 0.5

        channels = torch.stack([sem, q_graph, schema_prior, source_prior], dim=1)
        valid_pair = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
        channels = channels * valid_pair.unsqueeze(1)
        return channels, sem

    def forward(
        self,
        query_input_ids: torch.Tensor,
        query_attention_mask: torch.Tensor,
        workspace_input_ids: torch.Tensor,
        workspace_attention_mask: torch.Tensor,
        node_mask: torch.Tensor,
        schema_prior: torch.Tensor,
        source_prior: torch.Tensor,
        query_token_type_ids: Optional[torch.Tensor] = None,
        workspace_token_type_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        bsz, num_nodes, seq_len = workspace_input_ids.shape
        query_emb = self.encoder.encode(query_input_ids, query_attention_mask, query_token_type_ids)
        flat_ids = workspace_input_ids.view(bsz * num_nodes, seq_len)
        flat_mask = workspace_attention_mask.view(bsz * num_nodes, seq_len)
        flat_tti = workspace_token_type_ids.view(bsz * num_nodes, seq_len) if workspace_token_type_ids is not None else None
        node_embs = self.encoder.encode(flat_ids, flat_mask, flat_tti).view(bsz, num_nodes, -1)

        query_emb = self.query_proj(query_emb)
        node_embs = self.node_proj(node_embs) * node_mask.unsqueeze(-1)

        channels, sem = self._build_channels(query_emb, node_embs, node_mask, schema_prior, source_prior)
        graph_adj, align_prior, aux = self.adj_builder(channels, node_mask)

        gat_attn = graph_adj
        h = node_embs
        for layer in self.gat_layers:
            h, gat_attn = layer(h, graph_adj, node_mask)

        q = query_emb.unsqueeze(1).expand_as(h)
        feats = torch.cat([h, q, torch.abs(h - q), h * q], dim=-1)
        logits = self.scorer(feats).squeeze(-1)
        logits = logits.masked_fill(node_mask <= 0, -1e9)
        return {
            "query_emb": query_emb,
            "node_embs": h,
            "logits": logits,
            "gat_attn_weights": gat_attn,
            "graph_adj": graph_adj,
            "sheet_sim_scores": sem if self.graph_mode == "gtn_lite" else align_prior,
            "channel_info": aux,
        }


class AgentSheetStage2Loss(nn.Module):
    def __init__(self, tau: float = 0.07, lambda_align: float = 0.1, lambda_subgraph: float = 0.05, bce_weight: float = 1.0) -> None:
        super().__init__()
        self.tau = tau
        self.lambda_align = lambda_align
        self.lambda_subgraph = lambda_subgraph
        self.bce_weight = bce_weight

    def infonce_loss(self, query_emb: torch.Tensor, node_embs: torch.Tensor, labels: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        query_emb = F.normalize(query_emb, dim=-1)
        node_embs = F.normalize(node_embs, dim=-1)
        logits = torch.einsum("bd,bnd->bn", query_emb, node_embs) / self.tau
        logits = logits.masked_fill(node_mask <= 0, -1e9)
        total = query_emb.new_tensor(0.0)
        count = 0
        for b in range(query_emb.size(0)):
            pos_mask = (labels[b] > 0.5) & (node_mask[b] > 0.5)
            if pos_mask.sum() == 0:
                continue
            denom = torch.logsumexp(logits[b], dim=0)
            pos_logits = logits[b][pos_mask]
            total = total + (-pos_logits + denom).mean()
            count += 1
        if count == 0:
            return total
        return total / count

    def alignment_loss(self, gat_attn_weights: torch.Tensor, sheet_sim_scores: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        valid_pair = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
        eye = torch.eye(gat_attn_weights.size(-1), device=gat_attn_weights.device).unsqueeze(0)
        valid_pair = valid_pair * (1 - eye)
        diff = (gat_attn_weights - sheet_sim_scores.detach()) ** 2
        return (diff * valid_pair).sum() / valid_pair.sum().clamp(min=1.0)

    def subgraph_regularization(self, gat_attn_weights: torch.Tensor, labels: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        pos = (labels > 0.5).float() * node_mask
        pos_pair = torch.einsum("bi,bj->bij", pos, pos)
        eye = torch.eye(gat_attn_weights.size(-1), device=gat_attn_weights.device).unsqueeze(0)
        pos_pair = pos_pair * (1 - eye)
        if pos_pair.sum() <= 0:
            return gat_attn_weights.new_tensor(0.0)
        return -((gat_attn_weights * pos_pair).sum() / pos_pair.sum().clamp(min=1.0))

    def bce_loss(self, logits: torch.Tensor, labels: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        return (loss * node_mask).sum() / node_mask.sum().clamp(min=1.0)

    def forward(self, outputs: Dict[str, torch.Tensor], labels: torch.Tensor, node_mask: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss_infonce = self.infonce_loss(outputs["query_emb"], outputs["node_embs"], labels, node_mask)
        loss_bce = self.bce_loss(outputs["logits"], labels, node_mask)
        loss_align = self.alignment_loss(outputs["gat_attn_weights"], outputs["sheet_sim_scores"], node_mask)
        loss_subgraph = self.subgraph_regularization(outputs["gat_attn_weights"], labels, node_mask)
        total = loss_infonce + self.bce_weight * loss_bce + self.lambda_align * loss_align + self.lambda_subgraph * loss_subgraph
        stats = {
            "loss_infonce": float(loss_infonce.detach().item()),
            "loss_bce": float(loss_bce.detach().item()),
            "loss_align": float(loss_align.detach().item()),
            "loss_subgraph": float(loss_subgraph.detach().item()),
        }
        return total, stats


class Trainer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = setup_distributed()
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.rank = get_rank()
        seed_everything(args.seed + self.rank)

        if not args.allow_download:
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["HF_DATASETS_OFFLINE"] = "1"
        else:
            os.environ["TRANSFORMERS_OFFLINE"] = "0"
            os.environ["HF_DATASETS_OFFLINE"] = "0"

        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=not args.allow_download)
        self.train_dataset = NWayDataset(
            data_dir=args.data_dir,
            split="nway_train",
            tokenizer=self.tokenizer,
            max_length=args.max_length,
            max_query_length=args.max_query_length,
            max_workspace_size=args.max_workspace_size,
            features_file=args.train_features_file,
            max_header_texts=args.max_header_texts,
            include_shape_feature=args.include_shape_feature,
            include_source_feature=args.include_source_feature,
        )
        self.eval_dataset = NWayDataset(
            data_dir=args.data_dir,
            split="nway_eval",
            tokenizer=self.tokenizer,
            max_length=args.max_length,
            max_query_length=args.max_query_length,
            max_workspace_size=args.max_workspace_size,
            features_file=args.eval_features_file,
            max_header_texts=args.max_header_texts,
            include_shape_feature=args.include_shape_feature,
            include_source_feature=args.include_source_feature,
        )
        self.train_serializer = self.train_dataset.serializer
        self.eval_serializer = self.eval_dataset.serializer

        model = Stage2GTNModel(
            model_name=args.model_name,
            graph_mode=args.graph_mode,
            local_files_only=not args.allow_download,
            embedding_strategy=args.embedding_strategy,
            use_layer_mix=args.use_layer_mix,
            use_extra_position_embedding=args.use_extra_position_embedding,
            position_embedding_scale=args.position_embedding_scale,
            max_length=max(args.max_length, args.max_query_length),
            normalize_embeddings=args.normalize_embeddings,
            graph_dropout=args.graph_dropout,
            num_gat_layers=args.num_gat_layers,
            freeze_backbone=args.freeze_backbone,
        ).to(self.device)
        if args.stage1_checkpoint:
            model.load_stage1_checkpoint(args.stage1_checkpoint)

        if torch.cuda.is_available() and is_dist():
            model = DDP(model, device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=False)
        elif is_dist():
            model = DDP(model)
        self.model = model
        self.loss_fn = AgentSheetStage2Loss(
            tau=args.tau,
            lambda_align=args.lambda_align,
            lambda_subgraph=args.lambda_subgraph,
            bce_weight=args.bce_weight,
        )
        self.run_dir = os.path.join(args.output_dir, args.run_name)
        self.ckpt_dir = os.path.join(self.run_dir, "checkpoints")
        self.tb_writer = None
        if is_main_process():
            os.makedirs(self.ckpt_dir, exist_ok=True)
            if args.use_tensorboard:
                tb_dir = os.path.join(args.tensorboard_logdir, args.run_name)
                os.makedirs(tb_dir, exist_ok=True)
                self.tb_writer = SummaryWriter(log_dir=tb_dir)

    @property
    def raw_model(self) -> Stage2GTNModel:
        return self.model.module if isinstance(self.model, DDP) else self.model

    def log(self, msg: str) -> None:
        if is_main_process():
            print(msg, flush=True)

    def collate_fn(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        out = {}
        tensor_keys = [
            "query_input_ids", "query_attention_mask", "query_token_type_ids",
            "workspace_input_ids", "workspace_attention_mask", "workspace_token_type_ids",
            "node_mask", "labels",
        ]
        for k in tensor_keys:
            out[k] = torch.stack([item[k] for item in batch], dim=0)
        out["sheet_ids"] = [item["sheet_ids"] for item in batch]
        return out

    def build_priors(self, sheet_ids_batch: List[List[str]], serializer: SheetTextSerializer) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz = len(sheet_ids_batch)
        n = len(sheet_ids_batch[0])
        schema = torch.zeros(bsz, n, n, dtype=torch.float)
        source = torch.zeros(bsz, n, n, dtype=torch.float)
        for b, ids in enumerate(sheet_ids_batch):
            header_sets = [serializer.header_set(sid) for sid in ids]
            sources = [serializer.source(sid) for sid in ids]
            shapes = [serializer.shape(sid) for sid in ids]
            for i in range(n):
                if not ids[i]:
                    continue
                for j in range(n):
                    if i == j or not ids[j]:
                        continue
                    hi, hj = header_sets[i], header_sets[j]
                    if hi or hj:
                        inter = len(hi & hj)
                        union = max(len(hi | hj), 1)
                        schema[b, i, j] = inter / union
                    same_source = 1.0 if sources[i] and sources[i] == sources[j] else 0.0
                    r1, c1 = shapes[i]
                    r2, c2 = shapes[j]
                    shape_sim = 1.0 / (1.0 + abs(r1 - r2) + abs(c1 - c2))
                    source[b, i, j] = max(same_source, shape_sim)
        return schema, source

    def prepare_data(self):
        train_sampler = DistributedSampler(self.train_dataset, shuffle=True) if is_dist() else None
        eval_sampler = DistributedSampler(self.eval_dataset, shuffle=False) if is_dist() else None
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            num_workers=self.args.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=self.collate_fn,
        )
        eval_loader = DataLoader(
            self.eval_dataset,
            batch_size=self.args.batch_size,
            sampler=eval_sampler,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=self.collate_fn,
        )
        return train_loader, eval_loader, train_sampler

    def _move_batch(self, batch: Dict[str, torch.Tensor], serializer: SheetTextSerializer) -> Dict[str, torch.Tensor]:
        schema_prior, source_prior = self.build_priors(batch["sheet_ids"], serializer)
        for k, v in list(batch.items()):
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=True)
        batch["schema_prior"] = schema_prior.to(self.device, non_blocking=True)
        batch["source_prior"] = source_prior.to(self.device, non_blocking=True)
        return batch

    def save_checkpoint(self, name: str, optimizer, scheduler, epoch: int, best_metric: float) -> None:
        if not is_main_process():
            return
        path = os.path.join(self.ckpt_dir, name)
        torch.save(
            {
                "state_dict": self.raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "best_metric": best_metric,
                "args": vars(self.args),
            },
            path,
        )

    def compute_metrics(self, logits: torch.Tensor, labels: torch.Tensor, node_mask: torch.Tensor) -> Dict[str, float]:
        probs = torch.sigmoid(logits)
        pred = (probs >= self.args.activation_threshold).float() * node_mask
        tp = ((pred == 1) & (labels == 1) & (node_mask == 1)).sum().item()
        fp = ((pred == 1) & (labels == 0) & (node_mask == 1)).sum().item()
        fn = ((pred == 0) & (labels == 1) & (node_mask == 1)).sum().item()
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)

        subset_acc_sum = 0.0
        recall1_sum = 0.0
        recall3_sum = 0.0
        bsz = logits.size(0)
        for b in range(bsz):
            valid_idx = (node_mask[b] > 0.5).nonzero(as_tuple=True)[0]
            if valid_idx.numel() == 0:
                continue
            p = pred[b, valid_idx]
            y = labels[b, valid_idx]
            subset_acc_sum += float(torch.equal(p.cpu(), y.cpu()))

            pos_idx = (y > 0.5).nonzero(as_tuple=True)[0]
            if pos_idx.numel() > 0:
                scores = probs[b, valid_idx]
                top1 = torch.topk(scores, k=min(1, scores.numel())).indices
                top3 = torch.topk(scores, k=min(3, scores.numel())).indices
                recall1_sum += float((y[top1] > 0.5).any().item())
                recall3_sum += float((y[top3] > 0.5).any().item())
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "subset_acc": subset_acc_sum / max(bsz, 1),
            "recall_at_1": recall1_sum / max(bsz, 1),
            "recall_at_3": recall3_sum / max(bsz, 1),
        }

    def evaluate(self, eval_loader) -> EvalMetrics:
        self.model.eval()
        total_loss = total_r1 = total_r3 = total_f1 = total_subset = 0.0
        steps = 0
        with torch.no_grad():
            for batch in eval_loader:
                batch = self._move_batch(batch, self.eval_serializer)
                outputs = self.model(
                    query_input_ids=batch["query_input_ids"],
                    query_attention_mask=batch["query_attention_mask"],
                    query_token_type_ids=batch["query_token_type_ids"],
                    workspace_input_ids=batch["workspace_input_ids"],
                    workspace_attention_mask=batch["workspace_attention_mask"],
                    workspace_token_type_ids=batch["workspace_token_type_ids"],
                    node_mask=batch["node_mask"],
                    schema_prior=batch["schema_prior"],
                    source_prior=batch["source_prior"],
                )
                loss, _ = self.loss_fn(outputs, batch["labels"], batch["node_mask"])
                metrics = self.compute_metrics(outputs["logits"], batch["labels"], batch["node_mask"])
                total_loss += loss.item()
                total_r1 += metrics["recall_at_1"]
                total_r3 += metrics["recall_at_3"]
                total_f1 += metrics["f1"]
                total_subset += metrics["subset_acc"]
                steps += 1
        avg_loss = total_loss / max(steps, 1)
        avg_r1 = total_r1 / max(steps, 1)
        avg_r3 = total_r3 / max(steps, 1)
        avg_f1 = total_f1 / max(steps, 1)
        avg_subset = total_subset / max(steps, 1)
        return EvalMetrics(
            loss=reduce_mean(avg_loss, self.device),
            recall_at_1=reduce_mean(avg_r1, self.device),
            recall_at_3=reduce_mean(avg_r3, self.device),
            f1=reduce_mean(avg_f1, self.device),
            subset_acc=reduce_mean(avg_subset, self.device),
        )

    def train(self) -> None:
        train_loader, eval_loader, train_sampler = self.prepare_data()
        self.log(f"Train size: {len(self.train_dataset)} | Eval size: {len(self.eval_dataset)} | Graph mode: {self.args.graph_mode}")

        optimizer = AdamW((p for p in self.model.parameters() if p.requires_grad), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        total_steps = len(train_loader) * self.args.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * self.args.warmup_ratio),
            num_training_steps=total_steps,
        )

        best_metric = -1.0
        global_step = 0
        for epoch in range(self.args.num_epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            self.model.train()
            epoch_loss = 0.0
            epoch_r3 = 0.0
            steps = 0
            t0 = time.time()
            for batch in train_loader:
                batch = self._move_batch(batch, self.train_serializer)
                outputs = self.model(
                    query_input_ids=batch["query_input_ids"],
                    query_attention_mask=batch["query_attention_mask"],
                    query_token_type_ids=batch["query_token_type_ids"],
                    workspace_input_ids=batch["workspace_input_ids"],
                    workspace_attention_mask=batch["workspace_attention_mask"],
                    workspace_token_type_ids=batch["workspace_token_type_ids"],
                    node_mask=batch["node_mask"],
                    schema_prior=batch["schema_prior"],
                    source_prior=batch["source_prior"],
                )
                loss, loss_stats = self.loss_fn(outputs, batch["labels"], batch["node_mask"])
                optimizer.zero_grad()
                loss.backward()
                if self.args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                optimizer.step()
                scheduler.step()

                metrics = self.compute_metrics(outputs["logits"], batch["labels"], batch["node_mask"])
                epoch_loss += loss.item()
                epoch_r3 += metrics["recall_at_3"]
                steps += 1
                if self.tb_writer is not None and is_main_process():
                    self.tb_writer.add_scalar("train/loss_step", loss.item(), global_step)
                    self.tb_writer.add_scalar("train/recall_at_3_step", metrics["recall_at_3"], global_step)
                    self.tb_writer.add_scalar("train/loss_infonce_step", loss_stats["loss_infonce"], global_step)
                    self.tb_writer.add_scalar("train/loss_align_step", loss_stats["loss_align"], global_step)
                    self.tb_writer.add_scalar("train/loss_subgraph_step", loss_stats["loss_subgraph"], global_step)
                global_step += 1

            train_loss = reduce_mean(epoch_loss / max(steps, 1), self.device)
            train_r3 = reduce_mean(epoch_r3 / max(steps, 1), self.device)
            eval_metrics = self.evaluate(eval_loader)
            elapsed = time.time() - t0
            self.log(
                f"Epoch {epoch + 1}/{self.args.num_epochs} | "
                f"train_loss={train_loss:.4f} train_r3={train_r3:.4f} | "
                f"eval_loss={eval_metrics.loss:.4f} eval_r1={eval_metrics.recall_at_1:.4f} "
                f"eval_r3={eval_metrics.recall_at_3:.4f} eval_f1={eval_metrics.f1:.4f} subset_acc={eval_metrics.subset_acc:.4f} | "
                f"time={elapsed:.1f}s"
            )
            if self.tb_writer is not None and is_main_process():
                self.tb_writer.add_scalar("epoch/train_loss", train_loss, epoch + 1)
                self.tb_writer.add_scalar("epoch/train_recall_at_3", train_r3, epoch + 1)
                self.tb_writer.add_scalar("epoch/eval_loss", eval_metrics.loss, epoch + 1)
                self.tb_writer.add_scalar("epoch/eval_recall_at_1", eval_metrics.recall_at_1, epoch + 1)
                self.tb_writer.add_scalar("epoch/eval_recall_at_3", eval_metrics.recall_at_3, epoch + 1)
                self.tb_writer.add_scalar("epoch/eval_f1", eval_metrics.f1, epoch + 1)
                self.tb_writer.add_scalar("epoch/eval_subset_acc", eval_metrics.subset_acc, epoch + 1)

            metric = eval_metrics.recall_at_3
            if metric > best_metric:
                best_metric = metric
                self.save_checkpoint("best_stage2.pt", optimizer, scheduler, epoch + 1, best_metric)
            self.save_checkpoint("last_stage2.pt", optimizer, scheduler, epoch + 1, best_metric)

        if self.tb_writer is not None:
            self.tb_writer.flush()
            self.tb_writer.close()
            self.tb_writer = None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage2 GTN-lite / Full-GTN training")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--model-name", default="local_models/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594")
    parser.add_argument("--stage1-checkpoint", default="")
    parser.add_argument("--graph-mode", choices=["gtn_lite", "full_gtn"], default="gtn_lite")
    parser.add_argument("--run-name", default="stage2-gtn-lite")
    parser.add_argument("--output-dir", default="experiments/stage2_runs")
    parser.add_argument("--tensorboard-logdir", default="runs/stage2_gtn")
    parser.add_argument("--use-tensorboard", action="store_true")
    parser.add_argument("--allow-download", action="store_true")

    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--activation-threshold", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-query-length", type=int, default=64)
    parser.add_argument("--max-workspace-size", type=int, default=10)
    parser.add_argument("--max-header-texts", type=int, default=12)
    parser.add_argument("--train-features-file", default="")
    parser.add_argument("--eval-features-file", default="")
    parser.add_argument("--include-shape-feature", action="store_true")
    parser.add_argument("--include-source-feature", action="store_true")

    parser.add_argument("--embedding-strategy", choices=["cls", "mean", "max", "cls_mean_concat", "mean_max_concat", "cls_mean_max_concat"], default="cls")
    parser.add_argument("--use-layer-mix", action="store_true")
    parser.add_argument("--use-extra-position-embedding", action="store_true")
    parser.add_argument("--position-embedding-scale", type=float, default=1.0)
    parser.add_argument("--normalize-embeddings", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--graph-dropout", type=float, default=0.1)
    parser.add_argument("--num-gat-layers", type=int, default=1)

    parser.add_argument("--tau", type=float, default=0.07)
    parser.add_argument("--lambda-align", type=float, default=0.1)
    parser.add_argument("--lambda-subgraph", type=float, default=0.05)
    parser.add_argument("--bce-weight", type=float, default=1.0)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    trainer = Trainer(args)
    try:
        trainer.train()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
