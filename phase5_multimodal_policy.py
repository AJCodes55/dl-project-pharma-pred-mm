#!/usr/bin/env python3
"""
Phase 5 multimodal custom policy (Price + Sentiment + FDA).
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class Phase5MultimodalExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Dict,
        price_hidden: int = 128,
        sent_hidden: int = 64,
        fda_hidden: int = 64,
        fusion_dim: int = 256,
        num_attention_heads: int = 4,
    ) -> None:
        super().__init__(observation_space, features_dim=fusion_dim)
        if not isinstance(observation_space, spaces.Dict):
            raise ValueError("Phase5MultimodalExtractor requires Dict observation space.")

        price_shape = observation_space["price_seq"].shape
        sent_shape = observation_space["sent_seq"].shape
        fda_shape = observation_space["fda_seq"].shape
        ctx_shape = observation_space["portfolio_ctx"].shape
        if price_shape is None or sent_shape is None or fda_shape is None or ctx_shape is None:
            raise ValueError("Observation shapes must be defined.")

        self.window = int(price_shape[0])
        self.num_assets = int(price_shape[1])
        self.price_feat_dim = int(price_shape[2])
        self.sent_feat_dim = int(sent_shape[2])
        self.fda_feat_dim = int(fda_shape[2])
        self.ctx_dim = int(ctx_shape[0])

        self.price_lstm = nn.LSTM(
            input_size=self.price_feat_dim,
            hidden_size=price_hidden,
            num_layers=2,
            batch_first=True,
        )
        self.sent_lstm = nn.LSTM(
            input_size=self.sent_feat_dim,
            hidden_size=sent_hidden,
            num_layers=1,
            batch_first=True,
        )
        self.fda_mlp = nn.Sequential(
            nn.Linear(self.fda_feat_dim, fda_hidden),
            nn.ReLU(),
            nn.Linear(fda_hidden, fda_hidden),
            nn.ReLU(),
        )

        self.q_proj = nn.Linear(price_hidden, sent_hidden)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=sent_hidden,
            num_heads=num_attention_heads,
            batch_first=True,
        )
        asset_concat_dim = price_hidden + sent_hidden + fda_hidden
        self.fusion_proj = nn.Sequential(
            nn.Linear(self.num_assets * asset_concat_dim + self.ctx_dim, fusion_dim),
            nn.ReLU(),
            nn.LayerNorm(fusion_dim),
        )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        price_seq = observations["price_seq"].float()
        sent_seq = observations["sent_seq"].float()
        fda_seq = observations["fda_seq"].float()
        ctx = observations["portfolio_ctx"].float()

        b = price_seq.shape[0]
        price_seq = price_seq.permute(0, 2, 1, 3).reshape(b * self.num_assets, self.window, self.price_feat_dim)
        sent_seq = sent_seq.permute(0, 2, 1, 3).reshape(b * self.num_assets, self.window, self.sent_feat_dim)
        fda_seq = fda_seq.permute(0, 2, 1, 3).reshape(b * self.num_assets, self.window, self.fda_feat_dim)

        price_out, _ = self.price_lstm(price_seq)
        sent_out, _ = self.sent_lstm(sent_seq)
        q = self.q_proj(price_out)
        attn_out, _ = self.cross_attention(q, sent_out, sent_out, need_weights=False)

        # FDA branch uses last step of the sequence.
        fda_last = fda_seq[:, -1, :]
        fda_emb = self.fda_mlp(fda_last)

        price_last = price_out[:, -1, :]
        attn_last = attn_out[:, -1, :]
        fused_asset = torch.cat([price_last, attn_last, fda_emb], dim=-1)
        fused_asset = fused_asset.reshape(b, self.num_assets * fused_asset.shape[-1])
        fused = torch.cat([fused_asset, ctx], dim=-1)
        return self.fusion_proj(fused)


def build_phase5_policy_kwargs() -> Dict:
    return {
        "features_extractor_class": Phase5MultimodalExtractor,
        "features_extractor_kwargs": {
            "price_hidden": 128,
            "sent_hidden": 64,
            "fda_hidden": 64,
            "fusion_dim": 256,
            "num_attention_heads": 4,
        },
        "net_arch": {"pi": [128, 64], "vf": [128, 64]},
    }
