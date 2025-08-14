#!/usr/bin/env python3

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

try:
	from .log_utils import configure_logging
	from .dataset import (
		LabelConfig,
		SnapshotDataset,
		class_weights,
		feature_list,
		load_snapshots,
		make_labels,
		time_split,
	)
	from .model import MultiTaskClassifier
except Exception:
	# Fallback for direct script execution
	import sys as _sys
	_script_dir = os.path.abspath(os.path.dirname(__file__))
	if _script_dir not in _sys.path:
		_sys.path.append(_script_dir)
	from log_utils import configure_logging  # type: ignore
	from dataset import (  # type: ignore
		LabelConfig,
		SnapshotDataset,
		class_weights,
		feature_list,
		load_snapshots,
		make_labels,
		time_split,
	)
	from model import MultiTaskClassifier  # type: ignore


def select_device(prefer: str) -> torch.device:
	prefer = prefer.lower()
	if prefer == "cuda":
		return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
	if prefer == "cpu":
		return torch.device("cpu")
	# auto
	return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def ensure_usable_device(device: torch.device) -> torch.device:
	"""If CUDA is present but kernels are unsupported, fall back to CPU."""
	if device.type != "cuda":
		return device
	try:
		_ = torch.empty(1, device=device) * 2.0  # simple CUDA op
		return device
	except Exception as e:
		logging.warning(f"CUDA appears unusable ({e}); falling back to CPU.")
		return torch.device("cpu")


def log_device_info(device: torch.device) -> None:
	if device.type == "cuda":
		gpu_name = torch.cuda.get_device_name(0)
		cap = torch.cuda.get_device_capability(0)
		logging.info(f"CUDA | GPU: {gpu_name} | capability: {cap} | CUDA version: {torch.version.cuda}")
	else:
		logging.warning("Using CPU. For GPU, install a CUDA-enabled torch wheel matching your driver.")


def build_loaders(df, features: List[str], batch_size: int, device: torch.device) -> Tuple[DataLoader, DataLoader, DataLoader]:
	train_df, val_df, test_df = time_split(df)
	label_cols = ["y3", "y5", "y10"]
	train_ds = SnapshotDataset(train_df, features, label_cols)
	val_ds = SnapshotDataset(val_df, features, label_cols)
	test_ds = SnapshotDataset(test_df, features, label_cols)

	nw = max(2, (os.cpu_count() or 4) - 2)
	pin = device.type == "cuda"
	common = dict(num_workers=nw, pin_memory=pin, persistent_workers=True, prefetch_factor=4)
	train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **common)
	val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **common)
	test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **common)
	logging.info(f"DataLoader: num_workers={nw}, pin_memory={pin}, persistent_workers=True, prefetch_factor=4")
	return train_loader, val_loader, test_loader


def compute_metrics(logits_list: List[torch.Tensor], labels: torch.Tensor) -> Dict[str, float]:
	metrics = {}
	for i, name in enumerate(["3m", "5m", "10m"]):
		pred = torch.argmax(logits_list[i], dim=1)
		acc = (pred == labels[:, i]).float().mean().item()
		metrics[f"acc_{name}"] = acc
	return metrics


def calibration_plot(probs: np.ndarray, labels: np.ndarray, title: str, out_path: Path) -> None:
	p = probs[:, 2]
	y = (labels == 2).astype(int)
	bins = np.linspace(0.0, 1.0, 11)
	inds = np.digitize(p, bins) - 1
	bin_centers, frac_pos = [], []
	for b in range(len(bins) - 1):
		mask = inds == b
		if mask.any():
			bin_centers.append((bins[b] + bins[b + 1]) / 2)
			frac_pos.append(y[mask].mean())
	import matplotlib.pyplot as plt
	plt.figure(figsize=(4, 4))
	plt.plot([0, 1], [0, 1], "k--", label="perfect")
	plt.plot(bin_centers, frac_pos, "o-", label="empirical")
	plt.xlabel("Predicted P(up)")
	plt.ylabel("Empirical P(up)")
	plt.title(title)
	plt.legend()
	plt.grid(True, alpha=0.3)
	plt.tight_layout()
	plt.savefig(out_path)
	plt.close()


def train_one_epoch(model, loader, optimizer, scaler, device, loss_fns, epoch: int, log_every: int = 200, grad_accum_steps: int = 1):
	model.train()
	total_loss = 0.0
	optimizer.zero_grad(set_to_none=True)
	for step, (xb, yb) in enumerate(loader, start=1):
		xb = xb.to(device, non_blocking=True)
		yb = yb.to(device, non_blocking=True)
		ctx = torch.amp.autocast("cuda", enabled=(device.type == "cuda"))
		with ctx:
			logits3, logits5, logits10 = model(xb)
			loss3 = loss_fns[0](logits3, yb[:, 0])
			loss5 = loss_fns[1](logits5, yb[:, 1])
			loss10 = loss_fns[2](logits10, yb[:, 2])
			loss = (loss3 + loss5 + loss10) / 3.0
			loss = loss / grad_accum_steps
		scaler.scale(loss).backward()
		if step % grad_accum_steps == 0:
			scaler.step(optimizer)
			scaler.update()
			optimizer.zero_grad(set_to_none=True)
		total_loss += loss.item() * xb.size(0) * grad_accum_steps
		if device.type == "cuda" and (step % log_every == 0):
			mem = torch.cuda.memory_allocated() / (1024**3)
			mem_max = torch.cuda.max_memory_allocated() / (1024**3)
			logging.info(f"Epoch {epoch} Step {step}: GPU mem (GB): current={mem:.2f}, max={mem_max:.2f}")
	# flush leftover grads if loop ended mid-accumulation (rare)
	if device.type == "cuda":
		torch.cuda.synchronize()
		mem = torch.cuda.memory_allocated() / (1024**3)
		mem_max = torch.cuda.max_memory_allocated() / (1024**3)
		logging.info(f"GPU mem (GB) after epoch {epoch}: current={mem:.2f}, max={mem_max:.2f}")
	return total_loss / len(loader.dataset)


def evaluate(model, loader, device, loss_fns) -> Tuple[float, Dict[str, float], List[np.ndarray], np.ndarray]:
	model.eval()
	total_loss = 0.0
	all_probs3, all_probs5, all_probs10 = [], [], []
	all_labels = []
	with torch.no_grad():
		for xb, yb in loader:
			xb = xb.to(device, non_blocking=True)
			yb = yb.to(device, non_blocking=True)
			logits3, logits5, logits10 = model(xb)
			loss3 = loss_fns[0](logits3, yb[:, 0])
			loss5 = loss_fns[1](logits5, yb[:, 1])
			loss10 = loss_fns[2](logits10, yb[:, 2])
			loss = (loss3 + loss5 + loss10) / 3.0
			total_loss += loss.item() * xb.size(0)
			all_probs3.append(torch.softmax(logits3, dim=1).cpu().numpy())
			all_probs5.append(torch.softmax(logits5, dim=1).cpu().numpy())
			all_probs10.append(torch.softmax(logits10, dim=1).cpu().numpy())
			all_labels.append(yb.cpu().numpy())
	avg_loss = total_loss / len(loader.dataset)
	probs = [np.concatenate(all_probs3), np.concatenate(all_probs5), np.concatenate(all_probs10)]
	labels = np.concatenate(all_labels)
	metrics = compute_metrics([torch.from_numpy(p) for p in probs], torch.from_numpy(labels))
	return avg_loss, metrics, probs, labels


def find_largest_batch(
	df,
	features: List[str],
	device: torch.device,
	base_batch: int,
	target_util: float,
	hidden_dim: int,
	depth: int,
	dropout: float,
) -> int:
	"""Try to find the largest batch size that fits in memory up to target_util fraction of VRAM."""
	if device.type != "cuda":
		return base_batch
		total_mem = torch.cuda.get_device_properties(0).total_memory
	# Start from the provided base batch; expand until OOM, then bisect
	low = max(1, base_batch)
	high = low
	label_cols = ["y3", "y5", "y10"]
	train_df, _, _ = time_split(df)
	ds = SnapshotDataset(train_df, features, label_cols)
	while True:
		try:
			loader = DataLoader(ds, batch_size=high, shuffle=True, num_workers=2, pin_memory=True)
			xb, yb = next(iter(loader))
			xb = xb.to(device, non_blocking=True)
			yb = yb.to(device, non_blocking=True)
			model = MultiTaskClassifier(input_dim=xb.shape[1], hidden_dim=hidden_dim, dropout_p=dropout, depth=depth).to(device)
			with torch.no_grad():
				_ = model(xb)
			if torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory > target_util:
				break
			high *= 2
			if high > 65536:
				break
		except RuntimeError as e:
			if "out of memory" in str(e).lower():
				break
			else:
				raise
	# Binary search between low and high
	def fits(bs: int) -> bool:
		try:
			loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=2, pin_memory=True)
			xb, yb = next(iter(loader))
			xb = xb.to(device, non_blocking=True)
			yb = yb.to(device, non_blocking=True)
			model = MultiTaskClassifier(input_dim=xb.shape[1], hidden_dim=hidden_dim, dropout_p=dropout, depth=depth).to(device)
			with torch.no_grad():
				_ = model(xb)
			return torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory <= target_util
		except RuntimeError as e:
			if "out of memory" in str(e).lower():
				return False
			else:
				raise
	best = low
	while low <= high:
		mid = (low + high) // 2
		if fits(mid):
			best = mid
			low = mid + 1
		else:
			high = mid - 1
	logging.info(f"Auto-selected batch size: {best} targeting ~{int(target_util*100)}% VRAM")
	return best


def main():
	parser = argparse.ArgumentParser(description="Train multi-horizon classifier for EURUSD snapshots")
	parser.add_argument("--input", type=str, default=os.path.join(os.path.dirname(__file__), "eurusd-snapshots.json"))
	parser.add_argument("--pip-value", type=float, default=0.0001)
	parser.add_argument("--pip-threshold-pips", type=float, default=1.0)
	parser.add_argument("--epochs", type=int, default=30)
	parser.add_argument("--batch-size", type=str, default="8192", help="Int or 'auto' to search max batch")
	parser.add_argument("--grad-accum-steps", type=int, default=1)
	parser.add_argument("--lr", type=float, default=1e-3)
	parser.add_argument("--weight-decay", type=float, default=1e-4)
	parser.add_argument("--no-amp", action="store_true")
	parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Force device selection")
	parser.add_argument("--allow-tf32", action="store_true", help="Enable TF32 on matmul & cudnn for Ampere+")
	parser.add_argument("--compile", action="store_true", help="Use torch.compile for the model (PyTorch 2.0+)")
	parser.add_argument("--model", type=str, default="base", choices=["small", "base", "xl"], help="Model preset")
	parser.add_argument("--hidden-dim", type=int, default=None, help="Override hidden dimension")
	parser.add_argument("--depth", type=int, default=None, help="Override number of residual blocks")
	parser.add_argument("--dropout", type=float, default=None, help="Override dropout probability")
	parser.add_argument("--warmup-epochs", type=int, default=2)
	args = parser.parse_args()

	configure_logging()
	device = select_device(args.device)
	log_device_info(device)
	device = ensure_usable_device(device)
	log_device_info(device)
	amp_enabled = (device.type == "cuda" and not args.no_amp)
	logging.info(f"AMP enabled: {amp_enabled}")
	if device.type == "cuda":
		torch.backends.cudnn.benchmark = True
		if args.allow_tf32:
			torch.backends.cuda.matmul.allow_tf32 = True
			torch.backends.cudnn.allow_tf32 = True

	# Load and label
	df = load_snapshots(args.input)
	df = make_labels(df, LabelConfig(pip_value=args.pip_value, pip_threshold_pips=args.pip_threshold_pips))
	features = feature_list(df)

	# Presets
	preset = args.model
	if preset == "small":
		default_hidden, default_depth, default_dropout = 128, 2, 0.2
	elif preset == "base":
		default_hidden, default_depth, default_dropout = 512, 6, 0.1
	else:  # xl
		default_hidden, default_depth, default_dropout = 1024, 8, 0.1
	hidden_dim = args.hidden_dim or default_hidden
	depth = args.depth or default_depth
	dropout = args.dropout if args.dropout is not None else default_dropout

	# Batch size selection (consider chosen model capacity)
	if args.batch_size.strip().lower() == "auto":
		bs_base = 4096
		batch_size = find_largest_batch(
			df,
			features,
			device,
			bs_base,
			target_util=0.8,
			hidden_dim=hidden_dim,
			depth=depth,
			dropout=dropout,
		)
	else:
		batch_size = int(args.batch_size)
	train_loader, val_loader, test_loader = build_loaders(df, features, batch_size, device)

	# Compute class weights
	train_df, _, _ = time_split(df)
	cw = class_weights(train_df)
	loss_fns = [
		nn.CrossEntropyLoss(weight=cw["y3"].to(device)),
		nn.CrossEntropyLoss(weight=cw["y5"].to(device)),
		nn.CrossEntropyLoss(weight=cw["y10"].to(device)),
	]

	# Model
	model = MultiTaskClassifier(input_dim=len(features), hidden_dim=hidden_dim, dropout_p=dropout, depth=depth).to(device)
	p_dev = next(model.parameters()).device
	logging.info(f"Model parameter device: {p_dev}; preset={preset}; hidden_dim={hidden_dim}; depth={depth}; dropout={dropout}")
	if args.compile and hasattr(torch, "compile"):
		logging.info("Compiling model with torch.compile()")
		model = torch.compile(model)  # type: ignore
	optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

	# LR scheduler with warmup + cosine
	def lr_lambda(current_epoch: int) -> float:
		if current_epoch < args.warmup_epochs:
			return float(current_epoch + 1) / float(max(1, args.warmup_epochs))
		progress = (current_epoch - args.warmup_epochs) / float(max(1, args.epochs - args.warmup_epochs))
		return 0.5 * (1.0 + np.cos(np.pi * progress))
	scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

	best_val = float("inf")
	best_state = None
	for epoch in range(1, args.epochs + 1):
		train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, loss_fns, epoch, grad_accum_steps=args.grad_accum_steps)
		val_loss, val_metrics, _, _ = evaluate(model, val_loader, device, loss_fns)
		scheduler.step()
		logging.info(f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | {val_metrics}")
		if val_loss < best_val:
			best_val = val_loss
			best_state = model.state_dict()

	# Restore best and evaluate on test
	if best_state is not None:
		model.load_state_dict(best_state)
		if device.type == "cuda":
			torch.cuda.empty_cache()

	test_loss, test_metrics, probs, labels = evaluate(model, test_loader, device, loss_fns)
	logging.info(f"Test loss={test_loss:.4f} | {test_metrics}")

	# Save artifacts
	out_dir = Path(os.path.join(os.path.dirname(__file__), "checkpoints"))
	out_dir.mkdir(parents=True, exist_ok=True)
	(torch.save(model.state_dict(), out_dir / "model.pt"))
	(Path(out_dir / "features.json").write_text(json.dumps(features)))
	config = {
		"pip_value": args.pip_value,
		"pip_threshold_pips": args.pip_threshold_pips,
		"epochs": args.epochs,
		"batch_size": batch_size,
		"lr": args.lr,
		"weight_decay": args.weight_decay,
		"device": str(device),
		"preset": preset,
		"hidden_dim": hidden_dim,
		"depth": depth,
		"dropout": dropout,
		"allow_tf32": bool(args.allow_tf32),
		"compiled": bool(args.compile),
	}
	(Path(out_dir / "config.json").write_text(json.dumps(config, indent=2)))

	# Calibration plots (validate on test)
	for i, name in enumerate(["3m", "5m", "10m"]):
		cal_path = out_dir / f"calibration_{name}.png"
		calibration_plot(probs[i], labels[:, i], f"Calibration P(up) {name}", cal_path)
		logging.info(f"Saved calibration plot: {cal_path}")


if __name__ == "__main__":
	main() 