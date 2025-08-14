#!/usr/bin/env python3

import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class LabelConfig:
	pip_value: float = 0.0001
	pip_threshold_pips: float = 1.0

	def threshold_abs(self) -> float:
		return self.pip_value * self.pip_threshold_pips


class SnapshotDataset(Dataset):
	def __init__(self, df: pd.DataFrame, feature_cols: List[str], label_cols: List[str]):
		self.feature_cols = feature_cols
		self.label_cols = label_cols
		# Precompute tensors to reduce Python overhead in __getitem__
		X_np = df[feature_cols].to_numpy(dtype=np.float32, copy=True)
		labels_raw = df[label_cols].astype(int).to_numpy(copy=True)
		labels_idx = (labels_raw + 1).astype(np.int64)  # map {-1,0,1} -> {0,1,2}
		self.X = torch.from_numpy(X_np).contiguous()
		self.Y = torch.from_numpy(labels_idx)

	def __len__(self) -> int:
		return self.X.shape[0]

	def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
		return self.X[idx], self.Y[idx]


def load_snapshots(path: str) -> pd.DataFrame:
	logging.info(f"Loading snapshots from: {path}")
	with open(path, "r", encoding="utf-8") as f:
		data = json.load(f)
	df = pd.DataFrame(data)
	logging.info(f"Loaded snapshots: {len(df)} rows; columns: {sorted(df.columns.tolist())}")
	return df


def make_labels(df: pd.DataFrame, cfg: LabelConfig) -> pd.DataFrame:
	# future closes
	df = df.copy()
	df["close_3"] = df["close"].shift(-3)
	df["close_5"] = df["close"].shift(-5)
	df["close_10"] = df["close"].shift(-10)

	thr = cfg.threshold_abs()
	logging.info(f"Labeling with threshold_abs={thr} (pip_value={cfg.pip_value}, threshold_pips={cfg.pip_threshold_pips})")

	def label_from_delta(delta: pd.Series) -> pd.Series:
		return np.where(delta > thr, 1, np.where(delta < -thr, -1, 0))

	df["y3"] = label_from_delta(df["close_3"] - df["close"]).astype(int)
	df["y5"] = label_from_delta(df["close_5"] - df["close"]).astype(int)
	df["y10"] = label_from_delta(df["close_10"] - df["close"]).astype(int)

	# Drop rows with NaNs at the tail due to shifting
	before = len(df)
	df = df.dropna(subset=["close_3", "close_5", "close_10"]).reset_index(drop=True)
	logging.info(f"Dropped {before - len(df)} tail rows due to horizon shifts")
	logging.info("Label mapping for training: {-1,0,1} -> class indices {0,1,2}")
	return df


def time_split(df: pd.DataFrame, train_ratio=0.7, val_ratio=0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	n = len(df)
	train_end = int(n * train_ratio)
	val_end = int(n * (train_ratio + val_ratio))
	train_df = df.iloc[:train_end].reset_index(drop=True)
	val_df = df.iloc[train_end:val_end].reset_index(drop=True)
	test_df = df.iloc[val_end:].reset_index(drop=True)
	logging.info(f"Split sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
	return train_df, val_df, test_df


def class_weights(train_df: pd.DataFrame) -> Dict[str, torch.Tensor]:
	weights = {}
	for col in ["y3", "y5", "y10"]:
		counts = train_df[col].value_counts().reindex([-1, 0, 1]).fillna(0).astype(int)
		# Inverse frequency weights (avoid zero by +1); order aligns with class indices 0,1,2 via mapping (-1,0,1)
		inv = 1.0 / (counts.values + 1)
		w = torch.tensor(inv / inv.sum() * 3.0, dtype=torch.float32)
		weights[col] = w
		logging.info(f"Class distribution {col}: {counts.to_dict()} | weights (for classes [-1,0,1]->[0,1,2]): {w.tolist()}")
	return weights


def feature_list(df: pd.DataFrame) -> List[str]:
	# Use all engineered numeric features except datetime and label columns
	exclude = {"datetime", "close_3", "close_5", "close_10", "y3", "y5", "y10"}
	features = [c for c in df.columns if c not in exclude]
	# Ensure ordering and only numeric
	features = [c for c in features if np.issubdtype(df[c].dtype, np.number)]
	logging.info(f"Using features: {features}")
	return features 