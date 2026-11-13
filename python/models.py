from __future__ import annotations

import torch
from torch import nn

from python.config import ModelConfig

MODEL_NAMES = ("StatOnlyLR", "DKT", "SAKT", "CausalSeqOnly", "HSF_Concat", "HSF_Gate")
ABLATIONS = {"none": (0, 0), "item_prior": (0, 3), "user_cum": (3, 7), "recent_state": (7, 10)}


def causal_mask(length: int, device: torch.device) -> torch.Tensor:
    return torch.ones(length, length, device=device, dtype=torch.bool).triu(1)  # [L, L]


def mask_statistics(stats: torch.Tensor, group: str) -> torch.Tensor:
    if group not in ABLATIONS:
        raise ValueError(f"Unknown ablation: {group}")
    masked = stats.clone()  # [B, L, 10]
    start, end = ABLATIONS[group]
    masked[:, :, start:end] = 0  # [B, L, end-start]
    return masked  # [B, L, 10]


class StatOnlyLR(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Sequential(nn.LayerNorm(12), nn.Linear(12, 1))

    def forward(self, items: torch.Tensor, concepts: torch.Tensor, labels: torch.Tensor,
                stats: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, None]:
        odds = torch.logit(stats[:, :, :2].clamp(0.001, 0.999))  # [B, L, 2]
        features = torch.cat((stats, odds), dim=-1)  # [B, L, 12]
        return self.linear(features).squeeze(-1), None  # [B, L], None


class InteractionEmbedding(nn.Module):
    def __init__(self, num_items: int, num_concepts: int, dim: int, max_length: int) -> None:
        super().__init__()
        self.item = nn.Embedding(num_items + 2, dim, padding_idx=0)
        self.concept = nn.Embedding(num_concepts + 2, dim, padding_idx=0)
        self.answer = nn.Embedding(3, dim, padding_idx=0)
        self.position = nn.Embedding(max_length, dim)

    def forward(self, items: torch.Tensor, concepts: torch.Tensor,
                labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        length = items.shape[1]
        if length > self.position.num_embeddings:
            raise ValueError("Sequence exceeds configured context window")
        zeros = torch.zeros_like(items[:, :1])  # [B, 1]
        prev_items = torch.cat((zeros, items[:, :-1]), dim=1)  # [B, L]
        prev_concepts = torch.cat((zeros, concepts[:, :-1]), dim=1)  # [B, L]
        answers = (labels + 1) * (items != 0)  # [B, L]
        prev_answers = torch.cat((zeros, answers[:, :-1]), dim=1)  # [B, L]
        positions = torch.arange(length, device=items.device)  # [L]
        history = self.item(prev_items) + self.concept(prev_concepts)  # [B, L, D]
        history = history + self.answer(prev_answers) + self.position(positions)  # [B, L, D]
        target = self.item(items) + self.concept(concepts)  # [B, L, D]
        padding = prev_items == 0  # [B, L]
        padding[:, 0] = False  # [B]; BOS is always attendable
        return history, target, padding  # [B, L, D], [B, L, D], [B, L]


def prediction_head(dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU(),
                         nn.Dropout(dropout), nn.Linear(dim, 1))


class DKT(nn.Module):
    def __init__(self, num_items: int, num_concepts: int, cfg: ModelConfig, max_length: int) -> None:
        super().__init__()
        dim = cfg.hidden_dim
        self.embedding = InteractionEmbedding(num_items, num_concepts, dim, max_length)
        self.rnn = nn.GRU(dim, dim, batch_first=True)
        self.dropout = nn.Dropout(cfg.dropout)
        self.output = nn.Linear(2 * dim, 1)

    def forward(self, items: torch.Tensor, concepts: torch.Tensor, labels: torch.Tensor,
                stats: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, None]:
        history, target, _ = self.embedding(items, concepts, labels)  # [B, L, D], [B, L, D], [B, L]
        hidden, _ = self.rnn(history)  # [B, L, D], [1, B, D]
        combined = torch.cat((self.dropout(hidden), target), dim=-1)  # [B, L, 2D]
        return self.output(combined).squeeze(-1), None  # [B, L], None


class SAKT(nn.Module):
    def __init__(self, num_items: int, num_concepts: int, cfg: ModelConfig, max_length: int) -> None:
        super().__init__()
        dim = cfg.hidden_dim
        self.embedding = InteractionEmbedding(num_items, num_concepts, dim, max_length)
        self.attention = nn.MultiheadAttention(dim, cfg.num_heads, cfg.dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(),
                                 nn.Dropout(cfg.dropout), nn.Linear(2 * dim, dim))
        self.output = nn.Linear(dim, 1)

    def forward(self, items: torch.Tensor, concepts: torch.Tensor, labels: torch.Tensor,
                stats: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, None]:
        history, target, padding = self.embedding(items, concepts, labels)  # [B, L, D], [B, L, D], [B, L]
        attention_mask = causal_mask(items.shape[1], items.device)  # [L, L]
        hidden, _ = self.attention(target, history, history, attn_mask=attention_mask,
                                   key_padding_mask=padding, need_weights=False)  # [B, L, D], None
        hidden = self.norm1(target + hidden)  # [B, L, D]
        hidden = self.norm2(hidden + self.ffn(hidden))  # [B, L, D]
        return self.output(hidden).squeeze(-1), None  # [B, L], None


class CausalSeqOnly(nn.Module):
    def __init__(self, num_items: int, num_concepts: int, cfg: ModelConfig, max_length: int) -> None:
        super().__init__()
        dim = cfg.hidden_dim
        self.embedding = InteractionEmbedding(num_items, num_concepts, dim, max_length)
        layer = nn.TransformerEncoderLayer(dim, cfg.num_heads, 4 * dim,
                                           cfg.dropout, activation="gelu", batch_first=True)
        self.backbone = nn.TransformerEncoder(layer, cfg.num_layers, enable_nested_tensor=False)
        self.pred_head = prediction_head(dim, cfg.dropout)

    def encode(self, items: torch.Tensor, concepts: torch.Tensor,
               labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        history, target, padding = self.embedding(items, concepts, labels)  # [B, L, D], [B, L, D], [B, L]
        attention_mask = causal_mask(items.shape[1], items.device)  # [L, L]
        hidden = self.backbone(history, mask=attention_mask, src_key_padding_mask=padding)  # [B, L, D]
        return hidden, target  # [B, L, D], [B, L, D]

    def forward(self, items: torch.Tensor, concepts: torch.Tensor, labels: torch.Tensor,
                stats: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, None]:
        hidden, target = self.encode(items, concepts, labels)  # [B, L, D], [B, L, D]
        combined = torch.cat((hidden, target), dim=-1)  # [B, L, 2D]
        return self.pred_head(combined).squeeze(-1), None  # [B, L], None


class ContinuousStatEncoder(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.LayerNorm(10), nn.Linear(10, dim), nn.GELU(),
                                      nn.Dropout(dropout), nn.Linear(dim, dim), nn.LayerNorm(dim))

    def forward(self, stats: torch.Tensor) -> torch.Tensor:
        return self.network(stats)  # [B, L, D]


class HSFFusion(CausalSeqOnly):
    def __init__(self, num_items: int, num_concepts: int, cfg: ModelConfig,
                 max_length: int, gated: bool, ablate_group: str = "none") -> None:
        super().__init__(num_items, num_concepts, cfg, max_length)
        if ablate_group not in ABLATIONS:
            raise ValueError("Unknown ablation")
        self.ablate_group = ablate_group
        self.gated = gated
        self.stat_encoder = ContinuousStatEncoder(cfg.hidden_dim, cfg.dropout)
        # Identical parameter dimensions and initialization for the two fusions.
        self.fusion = nn.Linear(2 * cfg.hidden_dim, cfg.hidden_dim)
        self.prior_scale = nn.Parameter(torch.tensor(0.25))  # scalar

    def forward(self, items: torch.Tensor, concepts: torch.Tensor, labels: torch.Tensor,
                stats: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        hidden, target = self.encode(items, concepts, labels)  # [B, L, D], [B, L, D]
        features = mask_statistics(stats, self.ablate_group)  # [B, L, 10]
        encoded_stats = self.stat_encoder(features)  # [B, L, D]
        fusion_input = torch.cat((hidden, encoded_stats), dim=-1)  # [B, L, 2D]
        projected = self.fusion(fusion_input)  # [B, L, D]
        gate = torch.sigmoid(projected) if self.gated else None  # [B, L, D] or None
        fused = gate * hidden + (1 - gate) * encoded_stats if gate is not None else projected  # [B, L, D]
        prediction_input = torch.cat((fused, target), dim=-1)  # [B, L, 2D]
        logits = self.pred_head(prediction_input).squeeze(-1)  # [B, L]
        if self.ablate_group != "item_prior":
            bias = torch.logit(features[:, :, 0].clamp(0.001, 0.999))  # [B, L]
            logits = logits + self.prior_scale * bias  # [B, L]
        return logits, gate  # [B, L], [B, L, D] or None


class HSF_Concat(HSFFusion):
    def __init__(self, num_items: int, num_concepts: int, cfg: ModelConfig,
                 max_length: int, ablate_group: str = "none") -> None:
        super().__init__(num_items, num_concepts, cfg, max_length, False, ablate_group)


class HSF_Gate(HSFFusion):
    def __init__(self, num_items: int, num_concepts: int, cfg: ModelConfig,
                 max_length: int, ablate_group: str = "none") -> None:
        super().__init__(num_items, num_concepts, cfg, max_length, True, ablate_group)


def build_model(name: str, num_items: int, num_concepts: int, cfg: ModelConfig,
                max_length: int, ablate_group: str = "none") -> nn.Module:
    if name == "StatOnlyLR":
        return StatOnlyLR()
    classes = {"DKT": DKT, "SAKT": SAKT, "CausalSeqOnly": CausalSeqOnly,
               "HSF_Concat": HSF_Concat, "HSF_Gate": HSF_Gate}
    if name not in classes:
        raise ValueError(f"Unknown model: {name}")
    if name.startswith("HSF_"):
        return classes[name](num_items, num_concepts, cfg, max_length, ablate_group)
    if ablate_group != "none":
        raise ValueError("Ablation is only defined for HSF fusion models")
    return classes[name](num_items, num_concepts, cfg, max_length)
