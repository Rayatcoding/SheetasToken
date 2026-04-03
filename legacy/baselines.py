"""
Baseline 对比模型
==================
实现三个 Baseline，对应论文 Table 1 的对比实验：

  BM25Retriever:        经典稀疏检索（无学习）
  BertCrossEncoder:     原始 transformer.py 的方案（序列化 + Cross-Encoder）
  SBERTBiEncoder:       标准 Bi-Encoder（不含列级解耦和跨表 Attention）

每个 Baseline 都实现相同的接口：
  model.retrieve(query, workspace_features, top_k) → List[int]
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


# =============================================================================
# Baseline 1: BM25 稀疏检索
# =============================================================================

class BM25Retriever:
    """
    BM25 基线：将 sheet 的所有列名拼接为文档，用 BM25 检索。
    无需训练，直接用于 zero-shot 评估。
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

    def _sheet_to_tokens(self, feature: Dict) -> List[str]:
        headers = feature.get("headers", [])
        tokens = []
        for h in headers:
            text = h.get("text", "") if isinstance(h, dict) else str(h)
            tokens.extend(text.lower().split())
        return tokens

    def _query_to_tokens(self, query: str) -> List[str]:
        return query.lower().split()

    def retrieve(
        self,
        query: str,
        workspace_features: List[Dict],
        top_k: int = 3,
    ) -> Tuple[List[int], List[float]]:
        """返回 top-k 表格的索引和 BM25 分数"""
        from collections import Counter
        import math

        docs = [self._sheet_to_tokens(f) for f in workspace_features]
        q_tokens = self._query_to_tokens(query)
        N = len(docs)

        # 计算 IDF
        df = {}
        for doc in docs:
            for t in set(doc):
                df[t] = df.get(t, 0) + 1
        idf = {t: math.log((N - df[t] + 0.5) / (df[t] + 0.5) + 1) for t in df}

        # 平均文档长度
        avg_dl = sum(len(d) for d in docs) / max(N, 1)

        scores = []
        for doc in docs:
            tf = Counter(doc)
            dl = len(doc)
            score = 0.0
            for t in q_tokens:
                if t in tf:
                    f = tf[t]
                    score += idf.get(t, 0) * (
                        f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / avg_dl))
                    )
            scores.append(score)

        ranked = sorted(range(N), key=lambda i: scores[i], reverse=True)
        return ranked[:top_k], [scores[i] for i in ranked[:top_k]]


# =============================================================================
# Baseline 2: BERT Cross-Encoder（原 transformer.py 方案）
# =============================================================================

class BertCrossEncoder(nn.Module):
    """
    原始方案：将 query + sheet 文本拼接后过 BERT，直接分类。
    对应论文中的 "Serialization Baseline"。
    
    局限性：
      - 无法独立缓存 sheet embedding
      - 序列化丢失表格结构信息
      - N 张表需要 N 次前向传播
    """

    def __init__(
        self,
        backbone_name: str = "bert-base-uncased",
        max_length: int = 512,
        local_files_only: bool = True,
    ):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            backbone_name, local_files_only=local_files_only
        )
        self.classifier = nn.Linear(self.backbone.config.hidden_size, 1)
        self.max_length = max_length

    def _sheet_to_text(self, feature: Dict, max_headers: int = 20) -> str:
        headers = feature.get("headers", [])
        texts = [h.get("text", "") if isinstance(h, dict) else str(h) for h in headers[:max_headers]]
        shape = f"rows={feature.get('num_rows', 0)} cols={feature.get('num_cols', 0)}"
        return f"[SHEET] {shape} | " + " | ".join(texts)

    def forward_pair(
        self,
        query: str,
        sheet_feature: Dict,
        tokenizer,
        device: torch.device,
    ) -> float:
        sheet_text = self._sheet_to_text(sheet_feature)
        enc = tokenizer(
            query, sheet_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        with torch.no_grad():
            out = self.backbone(
                input_ids=enc["input_ids"].to(device),
                attention_mask=enc["attention_mask"].to(device),
                return_dict=True,
            )
            score = self.classifier(out.last_hidden_state[:, 0, :]).squeeze(-1)
        return float(torch.sigmoid(score).item())

    def retrieve(
        self,
        query: str,
        workspace_features: List[Dict],
        tokenizer,
        device: torch.device,
        top_k: int = 3,
    ) -> Tuple[List[int], List[float]]:
        scores = [
            self.forward_pair(query, f, tokenizer, device)
            for f in workspace_features
        ]
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return ranked[:top_k], [scores[i] for i in ranked[:top_k]]


# =============================================================================
# Baseline 3: SBERT Bi-Encoder（标准双塔，无列级解耦）
# =============================================================================

class SBERTBiEncoder(nn.Module):
    """
    标准 Bi-Encoder 基线：query 和 sheet 各自编码为全局 embedding，
    用余弦相似度检索。
    
    与 AgentSheet 的区别：
      - 无列级解耦（整张表序列化为一个文本）
      - 无 Value 统计特征
      - 无跨表 Attention
      - 无图结构
    
    这是消融实验中 "w/o Column-Level Encoding" 的对应模型。
    """

    def __init__(
        self,
        backbone_name: str = "bert-base-uncased",
        hidden_dim: int = 256,
        max_length: int = 256,
        local_files_only: bool = True,
    ):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            backbone_name, local_files_only=local_files_only
        )
        bert_dim = self.backbone.config.hidden_size
        self.proj = nn.Sequential(
            nn.Linear(bert_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.max_length = max_length

    def _sheet_to_text(self, feature: Dict, max_headers: int = 20) -> str:
        headers = feature.get("headers", [])
        texts = [h.get("text", "") if isinstance(h, dict) else str(h) for h in headers[:max_headers]]
        return " [SEP] ".join(texts)

    def encode(
        self,
        texts: List[str],
        tokenizer,
        device: torch.device,
        batch_size: int = 32,
    ) -> torch.Tensor:
        """批量编码文本列表，返回 L2 归一化的 embedding"""
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            enc = tokenizer(
                batch_texts,
                max_length=self.max_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            with torch.no_grad():
                out = self.backbone(
                    input_ids=enc["input_ids"].to(device),
                    attention_mask=enc["attention_mask"].to(device),
                    return_dict=True,
                )
                # Mean pooling
                mask = enc["attention_mask"].to(device).unsqueeze(-1).float()
                emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                emb = self.proj(emb)
                emb = F.normalize(emb, dim=-1)
            all_embs.append(emb.cpu())
        return torch.cat(all_embs, dim=0)

    def retrieve(
        self,
        query: str,
        workspace_features: List[Dict],
        tokenizer,
        device: torch.device,
        top_k: int = 3,
    ) -> Tuple[List[int], List[float]]:
        sheet_texts = [self._sheet_to_text(f) for f in workspace_features]
        all_texts = [query] + sheet_texts
        embs = self.encode(all_texts, tokenizer, device)

        q_emb = embs[0:1]       # (1, D)
        s_embs = embs[1:]        # (N, D)
        scores = (q_emb @ s_embs.T).squeeze(0).tolist()  # (N,)

        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return ranked[:top_k], [scores[i] for i in ranked[:top_k]]
