#!/usr/bin/env python3

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

try:
	from .log_utils import configure_logging
	from .model import MultiTaskClassifier
except Exception:
	# Fallback for direct script execution
	import sys as _sys
	_script_dir = os.path.abspath(os.path.dirname(__file__))
	if _script_dir not in _sys.path:
		_sys.path.append(_script_dir)
	from log_utils import configure_logging  # type: ignore
	from model import MultiTaskClassifier  # type: ignore


def load_features(path: Path) -> List[str]:
	return json.loads(path.read_text())


def predict_one(model, feature_names: List[str], sample: Dict) -> Dict:
	x = np.array([float(sample.get(k, 0.0)) for k in feature_names], dtype=np.float32)
	t = torch.from_numpy(x).unsqueeze(0)
	model.eval()
	with torch.no_grad():
		l3, l5, l10 = model(t)
		p3, p5, p10 = torch.softmax(l3, dim=1), torch.softmax(l5, dim=1), torch.softmax(l10, dim=1)
		c3, c5, c10 = torch.argmax(p3, dim=1).item(), torch.argmax(p5, dim=1).item(), torch.argmax(p10, dim=1).item()
		# Map 0/1/2 -> -1/0/1
		map_cls = {0: -1, 1: 0, 2: 1}
		return {
			"3m": {"probs": p3.squeeze(0).tolist(), "class": map_cls[c3]},
			"5m": {"probs": p5.squeeze(0).tolist(), "class": map_cls[c5]},
			"10m": {"probs": p10.squeeze(0).tolist(), "class": map_cls[c10]},
		}


def main():
	parser = argparse.ArgumentParser(description="Predict -1/0/1 probabilities for a snapshot")
	parser.add_argument("--model-dir", type=str, default=os.path.join(os.path.dirname(__file__), "checkpoints"))
	parser.add_argument("--input", type=str, required=True, help="Path to a JSON object or JSON file with one record")
	args = parser.parse_args()

	configure_logging()
	ckpt_dir = Path(args.model_dir)
	features = load_features(ckpt_dir / "features.json")

	state = torch.load(ckpt_dir / "model.pt", map_location="cpu")
	model = MultiTaskClassifier(input_dim=len(features))
	model.load_state_dict(state)

	# Load input (either a path to one JSON object file or a JSON object string)
	if os.path.isfile(args.input):
		obj = json.loads(Path(args.input).read_text())
	else:
		obj = json.loads(args.input)

	out = predict_one(model, features, obj)
	print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
	main() 