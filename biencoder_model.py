
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage1 Bi-Encoder training with DDP support.

Recommended placement:
repo_root/biencoder_model.py

Expected data files under --data-dir:
- pairwise_train.json
- pairwise_eval.json
- sheet_features_train.json / sheet_features_eval.json (or sheet_features.json)

Example:
torchrun --nproc_per_node=2 biencoder_model.py \
  --data-dir data \
  --use-tensorboard \
  --num-epochs 20
"""

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

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


def reduce_scalar(value: float, device: torch.device, average: bool = True) -> float:
    if not is_dist():
        return float(value)
    tensor = torch.tensor(float(value), dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    if average:
        tensor /= get_world_size()
    return tensor.item()


@dataclass
class EvalMetrics:
    loss: float
    accuracy: float
    precision: float
    recall: float
    f1: float


class SheetSimilarityDataset(Dataset):
    """Pairwise sheet similarity dataset for Stage1."""

    def __init__(
        self,
        data_dir: str,
        split: str,
        tokenizer,
        max_length: int = 256,
        sample_ratio: float = 1.0,
        max_samples: int = 0,
        stratified_sample: bool = False,
        random_seed: int = 42,
        features_file: Optional[str] = None,
        max_header_texts: int = 12,
        include_shape_feature: bool = True,
        include_source_feature: bool = True,
    ) -> None:
        self.data_dir = data_dir
        self.split = split
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sample_ratio = sample_ratio
        self.max_samples = max_samples
        self.stratified_sample = stratified_sample
        self.random_seed = random_seed
        self.features_file = features_file
        self.max_header_texts = max_header_texts
        self.include_shape_feature = include_shape_feature
        self.include_source_feature = include_source_feature

        self.sheet_feature_map = self._load_sheet_feature_map()
        self.data = self._sample_data(self._load_data())

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
        if split_name.startswith("pairwise_"):
            split_name = split_name[len("pairwise_"):]
        candidate = os.path.join(self.data_dir, f"sheet_features_{split_name}.json")
        if os.path.exists(candidate):
            return candidate
        fallback = os.path.join(self.data_dir, "sheet_features.json")
        if os.path.exists(fallback):
            return fallback
        return None

    def _load_sheet_feature_map(self) -> Dict[str, Dict]:
        feature_path = self._infer_features_file()
        if not feature_path:
            return {}
        with open(feature_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        feature_map = {}
        for item in raw:
            sheet_id = item.get("sheet_id")
            if sheet_id:
                feature_map[sheet_id] = item
        return feature_map

    def _sheet_feature_to_text(self, sheet_id: str) -> str:
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
            txt = str(h.get("text", "")).strip()
            if txt:
                header_texts.append(txt)
        if header_texts:
            segments.append("headers: " + " | ".join(header_texts))

        return " ; ".join(segments) if segments else str(sheet_id)

    def _sample_data(self, data: List[Dict]) -> List[Dict]:
        if not data:
            return data

        total = len(data)
        ratio = min(max(self.sample_ratio, 0.0), 1.0)
        target = total
        if ratio < 1.0:
            target = max(1, int(total * ratio))
        if self.max_samples and self.max_samples > 0:
            target = min(target, self.max_samples)

        if target >= total:
            return data

        rng = random.Random(self.random_seed)
        if not self.stratified_sample:
            sampled = rng.sample(data, target)
            rng.shuffle(sampled)
            return sampled

        label_to_items: Dict[int, List[Dict]] = {}
        for item in data:
            label = int(item.get("label", 0))
            label_to_items.setdefault(label, []).append(item)

        sampled: List[Dict] = []
        labels = sorted(label_to_items.keys())
        remaining = target
        remaining_classes = len(labels)
        for label in labels:
            group = label_to_items[label]
            quota = max(1, remaining // max(remaining_classes, 1))
            pick = min(len(group), quota)
            sampled.extend(rng.sample(group, pick))
            remaining -= pick
            remaining_classes -= 1

        if len(sampled) < target:
            sampled_ids = {id(x) for x in sampled}
            left = [x for x in data if id(x) not in sampled_ids]
            need = target - len(sampled)
            sampled.extend(rng.sample(left, min(need, len(left))))

        rng.shuffle(sampled)
        return sampled

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]

        text1 = item.get("sheet1_text", "")
        text2 = item.get("sheet2_text", "")

        if (not text1 or not text2) and ("sheet_a" in item and "sheet_b" in item):
            text1 = self._sheet_feature_to_text(item.get("sheet_a", ""))
            text2 = self._sheet_feature_to_text(item.get("sheet_b", ""))

        label = int(item.get("label", 0))

        enc1 = self.tokenizer(
            text1,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        enc2 = self.tokenizer(
            text2,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        sample = {
            "input_ids1": enc1["input_ids"].squeeze(0),
            "attention_mask1": enc1["attention_mask"].squeeze(0),
            "input_ids2": enc2["input_ids"].squeeze(0),
            "attention_mask2": enc2["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }
        if "token_type_ids" in enc1:
            sample["token_type_ids1"] = enc1["token_type_ids"].squeeze(0)
        if "token_type_ids" in enc2:
            sample["token_type_ids2"] = enc2["token_type_ids"].squeeze(0)
        return sample


class SimilarityClassifier(nn.Module):
    """Bi-Encoder similarity classifier."""

    def __init__(
        self,
        model_name: str,
        num_labels: int = 2,
        local_files_only: bool = True,
        embedding_strategy: str = "cls",
        use_layer_mix: bool = False,
        embedding_dropout: float = 0.1,
        head_hidden_dim: int = 0,
        use_extra_position_embedding: bool = False,
        position_embedding_scale: float = 1.0,
        max_length: int = 256,
        normalize_embeddings: bool = False,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.num_labels = num_labels
        self.embedding_strategy = embedding_strategy
        self.use_layer_mix = use_layer_mix
        self.use_extra_position_embedding = use_extra_position_embedding
        self.position_embedding_scale = position_embedding_scale
        self.normalize_embeddings = normalize_embeddings

        self.backbone = AutoModel.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
        hidden_size = self.backbone.config.hidden_size
        self.embedding_dim = self._get_embedding_dim(hidden_size)

        if self.use_extra_position_embedding:
            self.extra_position_embedding = nn.Embedding(max_length, hidden_size)

        if self.use_layer_mix:
            num_layers = self.backbone.config.num_hidden_layers + 1
            self.layer_weights = nn.Parameter(torch.zeros(num_layers))

        combined_dim = self.embedding_dim * 3
        self.dropout = nn.Dropout(embedding_dropout)
        if head_hidden_dim and head_hidden_dim > 0:
            self.classifier = nn.Sequential(
                nn.Linear(combined_dim, head_hidden_dim),
                nn.ReLU(),
                nn.Dropout(embedding_dropout),
                nn.Linear(head_hidden_dim, num_labels),
            )
        else:
            self.classifier = nn.Linear(combined_dim, num_labels)

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

    def forward(
        self,
        input_ids1: torch.Tensor,
        attention_mask1: torch.Tensor,
        input_ids2: torch.Tensor,
        attention_mask2: torch.Tensor,
        token_type_ids1: Optional[torch.Tensor] = None,
        token_type_ids2: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        emb1 = self.encode(input_ids1, attention_mask1, token_type_ids1)
        emb2 = self.encode(input_ids2, attention_mask2, token_type_ids2)
        diff = torch.abs(emb1 - emb2)
        combined = torch.cat([emb1, emb2, diff], dim=-1)
        logits = self.classifier(self.dropout(combined))
        cosine = F.cosine_similarity(emb1, emb2, dim=-1)
        return {"logits": logits, "emb1": emb1, "emb2": emb2, "cosine": cosine}


class Trainer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = setup_distributed()
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.rank = get_rank()
        self.world_size = get_world_size()

        seed_everything(args.seed + self.rank)

        if not args.allow_download:
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["HF_DATASETS_OFFLINE"] = "1"
        else:
            os.environ["TRANSFORMERS_OFFLINE"] = "0"
            os.environ["HF_DATASETS_OFFLINE"] = "0"

        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model_name,
            local_files_only=not args.allow_download,
        )

        model = SimilarityClassifier(
            model_name=args.model_name,
            num_labels=args.num_labels,
            local_files_only=not args.allow_download,
            embedding_strategy=args.embedding_strategy,
            use_layer_mix=args.use_layer_mix,
            embedding_dropout=args.embedding_dropout,
            head_hidden_dim=args.head_hidden_dim,
            use_extra_position_embedding=args.use_extra_position_embedding,
            position_embedding_scale=args.position_embedding_scale,
            max_length=args.max_length,
            normalize_embeddings=args.normalize_embeddings,
        ).to(self.device)

        if torch.cuda.is_available() and is_dist():
            model = DDP(model, device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=False)
        elif is_dist():
            model = DDP(model)

        self.model = model
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

        self.tb_writer = None
        self.run_dir = os.path.join(args.output_dir, args.run_name)
        self.ckpt_dir = os.path.join(self.run_dir, "checkpoints")
        if is_main_process():
            os.makedirs(self.ckpt_dir, exist_ok=True)
            if args.use_tensorboard:
                tb_dir = os.path.join(args.tensorboard_logdir, args.run_name)
                os.makedirs(tb_dir, exist_ok=True)
                self.tb_writer = SummaryWriter(log_dir=tb_dir)

    @property
    def raw_model(self) -> SimilarityClassifier:
        return self.model.module if isinstance(self.model, DDP) else self.model

    def log(self, msg: str) -> None:
        if is_main_process():
            print(msg, flush=True)

    def prepare_data(self):
        train_dataset = SheetSimilarityDataset(
            data_dir=self.args.data_dir,
            split="pairwise_train",
            tokenizer=self.tokenizer,
            max_length=self.args.max_length,
            sample_ratio=self.args.train_sample_ratio,
            max_samples=self.args.train_max_samples,
            stratified_sample=self.args.train_stratified_sample,
            random_seed=self.args.seed,
            features_file=self.args.train_features_file,
            max_header_texts=self.args.max_header_texts,
            include_shape_feature=self.args.include_shape_feature,
            include_source_feature=self.args.include_source_feature,
        )
        eval_dataset = SheetSimilarityDataset(
            data_dir=self.args.data_dir,
            split="pairwise_eval",
            tokenizer=self.tokenizer,
            max_length=self.args.max_length,
            sample_ratio=self.args.eval_sample_ratio,
            max_samples=self.args.eval_max_samples,
            stratified_sample=self.args.eval_stratified_sample,
            random_seed=self.args.seed,
            features_file=self.args.eval_features_file,
            max_header_texts=self.args.max_header_texts,
            include_shape_feature=self.args.include_shape_feature,
            include_source_feature=self.args.include_source_feature,
        )

        train_sampler = DistributedSampler(train_dataset, shuffle=True) if is_dist() else None
        eval_sampler = DistributedSampler(eval_dataset, shuffle=False) if is_dist() else None

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=self.args.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            sampler=eval_sampler,
            num_workers=self.args.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )
        return train_loader, eval_loader, train_sampler

    def create_optimizer(self):
        optimizer = AdamW(
            (p for p in self.model.parameters() if p.requires_grad),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )
        return optimizer

    def move_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

    def train_epoch(self, train_loader, train_sampler, optimizer, scheduler, epoch: int, global_step: int):
        self.model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        running_loss = 0.0
        running_correct = 0
        running_total = 0
        epoch_start = time.time()

        for step, batch in enumerate(train_loader):
            batch = self.move_batch(batch)
            labels = batch["labels"]

            outputs = self.model(
                input_ids1=batch["input_ids1"],
                attention_mask1=batch["attention_mask1"],
                input_ids2=batch["input_ids2"],
                attention_mask2=batch["attention_mask2"],
                token_type_ids1=batch.get("token_type_ids1"),
                token_type_ids2=batch.get("token_type_ids2"),
            )
            logits = outputs["logits"]
            loss = self.loss_fn(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            optimizer.step()
            scheduler.step()

            preds = torch.argmax(logits, dim=-1)
            running_loss += loss.item() * labels.size(0)
            running_correct += (preds == labels).sum().item()
            running_total += labels.size(0)

            if self.tb_writer is not None and is_main_process():
                self.tb_writer.add_scalar("train/loss_step", loss.item(), global_step)
                self.tb_writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
            global_step += 1

        train_loss = running_loss / max(running_total, 1)
        train_acc = running_correct / max(running_total, 1)

        train_loss = reduce_scalar(train_loss, self.device, average=True)
        train_acc = reduce_scalar(train_acc, self.device, average=True)
        epoch_time = time.time() - epoch_start
        epoch_time = reduce_scalar(epoch_time, self.device, average=True)
        return train_loss, train_acc, epoch_time, global_step

    @torch.no_grad()
    def evaluate(self, eval_loader) -> EvalMetrics:
        self.model.eval()

        total_loss = 0.0
        total = 0
        tp = tn = fp = fn = 0

        for batch in eval_loader:
            batch = self.move_batch(batch)
            labels = batch["labels"]

            outputs = self.model(
                input_ids1=batch["input_ids1"],
                attention_mask1=batch["attention_mask1"],
                input_ids2=batch["input_ids2"],
                attention_mask2=batch["attention_mask2"],
                token_type_ids1=batch.get("token_type_ids1"),
                token_type_ids2=batch.get("token_type_ids2"),
            )
            logits = outputs["logits"]
            loss = self.loss_fn(logits, labels)
            preds = torch.argmax(logits, dim=-1)

            total_loss += loss.item() * labels.size(0)
            total += labels.size(0)

            tp += ((preds == 1) & (labels == 1)).sum().item()
            tn += ((preds == 0) & (labels == 0)).sum().item()
            fp += ((preds == 1) & (labels == 0)).sum().item()
            fn += ((preds == 0) & (labels == 1)).sum().item()

        if is_dist():
            stats = torch.tensor([total_loss, total, tp, tn, fp, fn], dtype=torch.float64, device=self.device)
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            total_loss, total, tp, tn, fp, fn = stats.tolist()

        total = max(int(total), 1)
        loss = float(total_loss) / total
        accuracy = (tp + tn) / total
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)

        return EvalMetrics(
            loss=loss,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
        )

    def save_checkpoint(self, name: str, metrics: Optional[EvalMetrics] = None, epoch: Optional[int] = None) -> None:
        if not is_main_process():
            return
        save_dir = os.path.join(self.ckpt_dir, name)
        os.makedirs(save_dir, exist_ok=True)
        self.raw_model.backbone.save_pretrained(os.path.join(save_dir, "backbone"))
        self.tokenizer.save_pretrained(save_dir)
        payload = {
            "state_dict": self.raw_model.state_dict(),
            "model_name": self.args.model_name,
            "num_labels": self.args.num_labels,
            "embedding_strategy": self.args.embedding_strategy,
            "use_layer_mix": self.args.use_layer_mix,
            "embedding_dropout": self.args.embedding_dropout,
            "head_hidden_dim": self.args.head_hidden_dim,
            "use_extra_position_embedding": self.args.use_extra_position_embedding,
            "position_embedding_scale": self.args.position_embedding_scale,
            "normalize_embeddings": self.args.normalize_embeddings,
            "epoch": epoch,
            "metrics": metrics.__dict__ if metrics else None,
            "args": vars(self.args),
        }
        torch.save(payload, os.path.join(save_dir, "classifier.pt"))

    def train(self):
        train_loader, eval_loader, train_sampler = self.prepare_data()
        if is_main_process():
            with open(os.path.join(self.run_dir, "train_args.json"), "w", encoding="utf-8") as f:
                json.dump(vars(self.args), f, ensure_ascii=False, indent=2)

        optimizer = self.create_optimizer()
        total_steps = len(train_loader) * self.args.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=total_steps,
        )

        self.log(f"Rank {self.rank} using device {self.device}")
        self.log(f"Train size: {len(train_loader.dataset)}")
        self.log(f"Eval size: {len(eval_loader.dataset)}")

        best_f1 = -1.0
        global_step = 0

        for epoch in range(self.args.num_epochs):
            train_loss, train_acc, epoch_time, global_step = self.train_epoch(
                train_loader, train_sampler, optimizer, scheduler, epoch, global_step
            )
            metrics = self.evaluate(eval_loader)

            if is_main_process():
                self.log(
                    f"[Epoch {epoch + 1}/{self.args.num_epochs}] "
                    f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                    f"eval_loss={metrics.loss:.4f} eval_acc={metrics.accuracy:.4f} "
                    f"eval_p={metrics.precision:.4f} eval_r={metrics.recall:.4f} eval_f1={metrics.f1:.4f} "
                    f"time={epoch_time:.2f}s"
                )
                if self.tb_writer is not None:
                    self.tb_writer.add_scalar("epoch/train_loss", train_loss, epoch + 1)
                    self.tb_writer.add_scalar("epoch/train_acc", train_acc, epoch + 1)
                    self.tb_writer.add_scalar("epoch/eval_loss", metrics.loss, epoch + 1)
                    self.tb_writer.add_scalar("epoch/eval_acc", metrics.accuracy, epoch + 1)
                    self.tb_writer.add_scalar("epoch/eval_precision", metrics.precision, epoch + 1)
                    self.tb_writer.add_scalar("epoch/eval_recall", metrics.recall, epoch + 1)
                    self.tb_writer.add_scalar("epoch/eval_f1", metrics.f1, epoch + 1)
                    self.tb_writer.add_scalar("epoch/time_sec", epoch_time, epoch + 1)
                    self.tb_writer.flush()

            if metrics.f1 > best_f1:
                best_f1 = metrics.f1
                self.save_checkpoint("best_model", metrics=metrics, epoch=epoch + 1)

        final_metrics = self.evaluate(eval_loader)
        self.save_checkpoint("final_model", metrics=final_metrics, epoch=self.args.num_epochs)
        self.log(f"Training done. Best eval F1 = {best_f1:.4f}")
        if self.tb_writer is not None:
            self.tb_writer.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage1 Bi-Encoder DDP trainer")

    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument(
        "--model-name",
        type=str,
        default="local_models/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594",
    )
    parser.add_argument("--output-dir", type=str, default="outputs/stage1_biencoder")
    parser.add_argument("--run-name", type=str, default="stage1_biencoder")

    parser.add_argument("--num-labels", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--embedding-strategy",
        choices=[
            "cls",
            "mean",
            "max",
            "cls_mean_concat",
            "mean_max_concat",
            "cls_mean_max_concat",
        ],
        default="cls",
    )
    parser.add_argument("--use-layer-mix", action="store_true")
    parser.add_argument("--embedding-dropout", type=float, default=0.1)
    parser.add_argument("--head-hidden-dim", type=int, default=0)
    parser.add_argument("--use-extra-position-embedding", action="store_true")
    parser.add_argument("--position-embedding-scale", type=float, default=1.0)
    parser.add_argument("--normalize-embeddings", action="store_true")

    parser.add_argument("--train-sample-ratio", type=float, default=1.0)
    parser.add_argument("--train-max-samples", type=int, default=0)
    parser.add_argument("--train-stratified-sample", action="store_true")
    parser.add_argument("--eval-sample-ratio", type=float, default=1.0)
    parser.add_argument("--eval-max-samples", type=int, default=0)
    parser.add_argument("--eval-stratified-sample", action="store_true")

    parser.add_argument("--train-features-file", type=str, default=None)
    parser.add_argument("--eval-features-file", type=str, default=None)
    parser.add_argument("--max-header-texts", type=int, default=12)
    parser.add_argument("--include-shape-feature", action="store_true")
    parser.add_argument("--include-source-feature", action="store_true")

    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--use-tensorboard", action="store_true")
    parser.add_argument("--tensorboard-logdir", type=str, default="runs/stage1_biencoder")
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    # Keep defaults aligned with original biencoder file: both enabled unless user explicitly removes flags.
    if "--include-shape-feature" not in os.sys.argv:
        args.include_shape_feature = True
    if "--include-source-feature" not in os.sys.argv:
        args.include_source_feature = True

    trainer = Trainer(args)
    try:
        trainer.train()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
