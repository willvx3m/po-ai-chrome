#!/usr/bin/env python3

import torch
import torch.nn as nn
from typing import Tuple


class ResidualMLPBlock(nn.Module):
	def __init__(self, hidden_dim: int, dropout_p: float = 0.1, activation: str = "gelu"):
		super().__init__()
		self.norm = nn.LayerNorm(hidden_dim)
		act_layer = nn.GELU if activation == "gelu" else nn.ReLU
		self.ff = nn.Sequential(
			nn.Linear(hidden_dim, hidden_dim * 4),
			act_layer(),
			nn.Dropout(p=dropout_p),
			nn.Linear(hidden_dim * 4, hidden_dim),
			nn.Dropout(p=dropout_p),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		z = self.norm(x)
		z = self.ff(z)
		return x + z


class MultiTaskClassifier(nn.Module):
	def __init__(self, input_dim: int, hidden_dim: int = 128, dropout_p: float = 0.2, depth: int = 2, activation: str = "gelu"):
		super().__init__()
		self.input_norm = nn.LayerNorm(input_dim)

		self.input_proj = nn.Sequential(
			nn.Linear(input_dim, hidden_dim),
			nn.GELU() if activation == "gelu" else nn.ReLU(inplace=True),
			nn.Dropout(p=dropout_p),
		)

		blocks = []
		for _ in range(max(1, depth)):
			blocks.append(ResidualMLPBlock(hidden_dim=hidden_dim, dropout_p=dropout_p, activation=activation))
		self.trunk = nn.Sequential(*blocks)

		# Three heads: for 3m, 5m, 10m; each outputs logits for 3 classes (-1,0,1)
		self.head_3m = nn.Linear(hidden_dim, 3)
		self.head_5m = nn.Linear(hidden_dim, 3)
		self.head_10m = nn.Linear(hidden_dim, 3)

	def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		# x: (B, input_dim)
		x = self.input_norm(x)
		z = self.input_proj(x)
		z = self.trunk(z)
		logits_3 = self.head_3m(z)
		logits_5 = self.head_5m(z)
		logits_10 = self.head_10m(z)
		return logits_3, logits_5, logits_10 