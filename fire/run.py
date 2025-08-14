#!/usr/bin/env python3
"""
Backtesting script that queries an OpenAI model at each timepoint for EURUSD snapshots.

- Loads dataset (default: ../dataset/eurusd-snapshots.json) containing vectors:
  [datetime, close, ema_fast_change_pct_3m, ema_slow_change_pct_6m, ema_spread_pct,
   price_to_ema_fast_pct, vol_std_10m_pct, macd_hist_pct, rsi14_centered,
   ret_log_5m_pct, ret_log_15m_pct]
- Computes realized log return percentages for 1/3/5/10-minute horizons
- Calls OpenAI model (default: gpt-4o-mini) per snapshot to obtain predictions
  (will_go_up, probability_up, expected_return_pct, rationale) for each horizon
- Caches responses to avoid re-calling for the same input
- Writes per-row results and cache under a timestamped results/ directory
  (e.g., results/results_YYYYMMDD_HHMMSS.csv and results/cache_YYYYMMDD_HHMMSS.jsonl)
- Saves all prompts used to results/prompts_YYYYMMDD_HHMMSS.jsonl
- Saves all model responses (raw + parsed summary) to results/responses_YYYYMMDD_HHMMSS.jsonl
- Prints + logs final metrics to log.txt

Run from the fire/ directory. Example:
    python3 run.py --model gpt-4o-mini --max-rows 1000 --stride 5

Requires OPENAI_API_KEY in environment.
"""

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
	from openai import OpenAI
	_HAS_OPENAI = True
except Exception:  # pragma: no cover
	_HAS_OPENAI = False
	OpenAI = None  # type: ignore


FEATURE_COLUMNS = [
	"datetime",
	"close",
	"ema_fast_change_pct_3m",
	"ema_slow_change_pct_6m",
	"ema_spread_pct",
	"price_to_ema_fast_pct",
	"vol_std_10m_pct",
	"macd_hist_pct",
	"rsi14_centered",
	"ret_log_5m_pct",
	"ret_log_15m_pct",
]

HORIZONS_MIN = [1, 3, 5, 10]


@dataclass
class ModelPrediction:
	will_go_up: bool
	probability_up: float
	expected_return_pct: float
	rationale: str


@dataclass
class RowPrediction:
	pred_h1m: Optional[ModelPrediction]
	pred_h3m: Optional[ModelPrediction]
	pred_h5m: Optional[ModelPrediction]
	pred_h10m: Optional[ModelPrediction]

	def to_row_dict(self) -> Dict[str, Any]:
		out: Dict[str, Any] = {}
		for label, pred in [
			("1m", self.pred_h1m),
			("3m", self.pred_h3m),
			("5m", self.pred_h5m),
			("10m", self.pred_h10m),
		]:
			if pred is None:
				out[f"pred_{label}_will_go_up"] = None
				out[f"pred_{label}_probability_up"] = None
				out[f"pred_{label}_expected_return_pct"] = None
				out[f"pred_{label}_rationale"] = None
			else:
				out[f"pred_{label}_will_go_up"] = bool(pred.will_go_up)
				out[f"pred_{label}_probability_up"] = float(pred.probability_up)
				out[f"pred_{label}_expected_return_pct"] = float(pred.expected_return_pct)
				out[f"pred_{label}_rationale"] = pred.rationale
		return out


class JsonlCache:
	"""Append-only JSONL cache mapping a deterministic key to the model response JSON."""

	def __init__(self, cache_path: Path) -> None:
		self.cache_path = cache_path
		self._memory: Dict[str, Dict[str, Any]] = {}
		self._loaded = False

	def _ensure_loaded(self) -> None:
		if self._loaded:
			return
		self._loaded = True
		if not self.cache_path.exists():
			return
		with self.cache_path.open("r", encoding="utf-8") as f:
			for line in f:
				line = line.strip()
				if not line:
					continue
				try:
					obj = json.loads(line)
					k = obj.get("key")
					v = obj.get("response")
					if isinstance(k, str) and v is not None:
						self._memory[k] = v
				except Exception:
					continue

	def get(self, key: str) -> Optional[Dict[str, Any]]:
		self._ensure_loaded()
		return self._memory.get(key)

	def set(self, key: str, response: Dict[str, Any]) -> None:
		self._ensure_loaded()
		self._memory[key] = response
		self.cache_path.parent.mkdir(parents=True, exist_ok=True)
		with self.cache_path.open("a", encoding="utf-8") as f:
			f.write(json.dumps({"key": key, "response": response}, ensure_ascii=False) + "\n")


def configure_logging(log_path: Path) -> None:
	logger = logging.getLogger()
	logger.setLevel(logging.INFO)
	logger.handlers = []

	fmt = logging.Formatter(
		"%(asctime)s | %(levelname)s | %(message)s",
		datefmt="%Y-%m-%d %H:%M:%S",
	)

	stream_handler = logging.StreamHandler(sys.stdout)
	stream_handler.setFormatter(fmt)
	logger.addHandler(stream_handler)

	file_handler = logging.FileHandler(log_path, encoding="utf-8")
	file_handler.setFormatter(fmt)
	logger.addHandler(file_handler)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="LLM backtest on EURUSD snapshots")
	parser.add_argument("--dataset", type=str, default="../dataset/eurusd-snapshots.json", help="Path to dataset JSON")
	parser.add_argument("--model", type=str, default=None, help="OpenAI model name (defaults to OPENAI_MODEL/MODEL/.env or gpt-4o-mini)")
	parser.add_argument("--window", type=int, default=10, help="Trailing window size (minutes) to include as context; 0 = none")
	parser.add_argument("--max-rows", type=int, default=None, help="Limit number of rows to process")
	parser.add_argument("--stride", type=int, default=1, help="Process every Nth row for cost control")
	parser.add_argument("--out-dir", type=str, default="results", help="Directory to store timestamped results and cache")
	parser.add_argument("--cache", type=str, default="cache.jsonl", help="Path to JSONL cache file")
	parser.add_argument("--rpm", type=float, default=200.0, help="Rate limit: requests per minute")
	parser.add_argument("--timeout", type=int, default=30, help="Request timeout seconds")
	parser.add_argument("--out", type=str, default="results.csv", help="Output CSV path")
	parser.add_argument("--prompts", type=str, default="prompts.jsonl", help="Path to prompts JSONL file")
	parser.add_argument("--responses", type=str, default="responses.jsonl", help="Path to responses JSONL file")
	parser.add_argument("--log", type=str, default="log.txt", help="Log file path")
	parser.add_argument("--p-threshold", type=float, default=0.55, help="Prob threshold for optional PnL proxy")
	parser.add_argument("--tx-cost", type=float, default=0.0, help="Transaction cost per trade (return pct units)")
	return parser.parse_args()


def load_env_file(dotenv_path: Path) -> bool:
	"""Load a simple .env (KEY=VALUE) file into os.environ. Returns True if loaded."""
	if not dotenv_path.exists():
		return False
	try:
		with dotenv_path.open("r", encoding="utf-8") as f:
			for raw in f:
				line = raw.strip()
				if not line or line.startswith("#"):
					continue
				if "=" not in line:
					continue
				key, value = line.split("=", 1)
				key = key.strip()
				if key.startswith("export "):
					key = key[len("export ") :].strip()
				val = value.strip()
				# Strip surrounding quotes
				if (val.startswith("\"") and val.endswith("\"")) or (val.startswith("'") and val.endswith("'")):
					val = val[1:-1]
				os.environ[key] = val
		return True
	except Exception:
		return False


def load_dataset(dataset_path: Path) -> pd.DataFrame:
	with dataset_path.open("r", encoding="utf-8") as f:
		data = json.load(f)

	if not isinstance(data, list) or len(data) == 0:
		raise ValueError("Dataset JSON must be a non-empty list")

	first = data[0]
	if isinstance(first, dict):
		# Normalize dicts to the expected columns order
		records: List[Dict[str, Any]] = []
		for obj in data:
			row: Dict[str, Any] = {}
			for col in FEATURE_COLUMNS:
				row[col] = obj.get(col)
			records.append(row)
		df = pd.DataFrame.from_records(records)
	elif isinstance(first, list) or isinstance(first, tuple):
		# Assume vectors align with FEATURE_COLUMNS
		df = pd.DataFrame(data, columns=FEATURE_COLUMNS[: len(first)])
		# If fewer columns provided, raise an error; if more, slice
		if df.shape[1] != len(FEATURE_COLUMNS):
			raise ValueError(f"Expected {len(FEATURE_COLUMNS)} columns, got {df.shape[1]}")
	else:
		raise ValueError("Dataset JSON must be a list of lists or list of dicts")

	# Coerce numeric types, keep datetime as string
	for col in FEATURE_COLUMNS:
		if col == "datetime":
			continue
		df[col] = pd.to_numeric(df[col], errors="coerce")

	# Drop rows with missing required fields
	required_cols = ["datetime", "close"]
	df = df.dropna(subset=required_cols).reset_index(drop=True)
	return df


def compute_realized_returns(df: pd.DataFrame) -> pd.DataFrame:
	result = df.copy()
	for h in HORIZONS_MIN:
		future_close = result["close"].shift(-h)
		realized = np.log(future_close / result["close"]) * 100.0
		result[f"realized_return_pct_{h}m"] = realized
		result[f"final_close_{h}m"] = future_close
	# Drop last max(H) rows that cannot be labeled
	max_h = max(HORIZONS_MIN)
	return result.iloc[:-max_h].reset_index(drop=True)


def build_prompt(single_snapshot: Dict[str, Any], trailing_context: Optional[List[Dict[str, Any]]] = None) -> str:
	lines: List[str] = []
	lines.append("You are a quantitative FX analyst. Do not use any future information beyond the provided snapshot(s).")
	lines.append("")
	lines.append("Feature definitions:")
	lines.append("- ema_fast_change_pct_3m: % change of EMA(6) over 3 minutes")
	lines.append("- ema_slow_change_pct_6m: % change of EMA(13) over 6 minutes")
	lines.append("- ema_spread_pct: % (EMA6 - EMA13) / EMA13")
	lines.append("- price_to_ema_fast_pct: % distance of close to EMA(6)")
	lines.append("- vol_std_10m_pct: 100 * std of 1m log-returns over 10 minutes (scaled x10)")
	lines.append("- macd_hist_pct: 100 * (MACD(6,13) - Signal(4)) / EMA(13) (scaled x10)")
	lines.append("- rsi14_centered: (RSI(7) - 50) / 50 in [-1, 1]")
	lines.append("- ret_log_5m_pct: 100 * ln(close/close[-5]) (scaled x10)")
	lines.append("- ret_log_15m_pct: 100 * ln(close/close[-15]) (scaled x10)")
	lines.append("")
	if trailing_context and len(trailing_context) > 0:
		lines.append("Trailing context (most recent last):")
		for ctx in trailing_context:
			lines.append(
				f"- {ctx['datetime']}: close={ctx['close']}, ema_fast_change_pct_3m={ctx['ema_fast_change_pct_3m']}, "
				f"ema_slow_change_pct_6m={ctx['ema_slow_change_pct_6m']}, ema_spread_pct={ctx['ema_spread_pct']}, "
				f"price_to_ema_fast_pct={ctx['price_to_ema_fast_pct']}, vol_std_10m_pct={ctx['vol_std_10m_pct']}, "
				f"macd_hist_pct={ctx['macd_hist_pct']}, rsi14_centered={ctx['rsi14_centered']}, "
				f"ret_log_5m_pct={ctx['ret_log_5m_pct']}, ret_log_15m_pct={ctx['ret_log_15m_pct']}"
			)
		lines.append("")

	lines.append("Current snapshot:")
	lines.append(
		f"- datetime: {single_snapshot['datetime']}\n"
		f"- close: {single_snapshot['close']}\n"
		f"- ema_fast_change_pct_3m: {single_snapshot['ema_fast_change_pct_3m']}\n"
		f"- ema_slow_change_pct_6m: {single_snapshot['ema_slow_change_pct_6m']}\n"
		f"- ema_spread_pct: {single_snapshot['ema_spread_pct']}\n"
		f"- price_to_ema_fast_pct: {single_snapshot['price_to_ema_fast_pct']}\n"
		f"- vol_std_10m_pct: {single_snapshot['vol_std_10m_pct']}\n"
		f"- macd_hist_pct: {single_snapshot['macd_hist_pct']}\n"
		f"- rsi14_centered: {single_snapshot['rsi14_centered']}\n"
		f"- ret_log_5m_pct: {single_snapshot['ret_log_5m_pct']}\n"
		f"- ret_log_15m_pct: {single_snapshot['ret_log_15m_pct']}"
	)
	lines.append("")
	lines.append("Task: For horizons 1m, 3m, 5m, and 10m, you must respond with a JSON object ONLY in the following format. Do not include any extra keys (like datetime, close, features) at the top level:")
	lines.append("")
	lines.append("Top-level JSON object schema (must match exactly):")
	lines.append("- predictions: object with keys '1m', '3m', '5m', '10m'")
	lines.append("  - Each horizon object must contain:")
	lines.append("    - will_go_up: boolean")
	lines.append("    - probability_up: number between 0 and 1")
	lines.append("    - expected_return_pct: number (percent; positive means up, negative means down)")
	lines.append("    - rationale: short string (<= 300 chars)")
	lines.append("")
	lines.append("Return JSON EXACTLY in this template (values are examples):")
	lines.append("{")
	lines.append("  \"predictions\": {")
	lines.append("    \"1m\": { \"will_go_up\": true,  \"probability_up\": 0.62, \"expected_return_pct\": 0.03, \"rationale\": \"...\" },")
	lines.append("    \"3m\": { \"will_go_up\": false, \"probability_up\": 0.41, \"expected_return_pct\": -0.05, \"rationale\": \"...\" },")
	lines.append("    \"5m\": { \"will_go_up\": true,  \"probability_up\": 0.58, \"expected_return_pct\": 0.07, \"rationale\": \"...\" },")
	lines.append("    \"10m\":{ \"will_go_up\": true,  \"probability_up\": 0.55, \"expected_return_pct\": 0.10, \"rationale\": \"...\" }")
	lines.append("  }")
	lines.append("}")
	lines.append("")
	lines.append("Constraints:")
	lines.append("- Do NOT add any other top-level keys besides 'predictions'.")
	lines.append("- Do NOT echo the input snapshot. Only output the predictions JSON above.")
	lines.append("- Use floats for numbers; probability_up in [0,1].")
	lines.append("- If uncertain, provide your best estimate; do not return null.")
	return "\n".join(lines)


def call_model(
	client: Any,
	model: str,
	prompt_text: str,
	timeout_sec: int,
	retry: int = 3,
	sleep_between_sec: float = 1.0,
) -> Tuple[Dict[str, Any], str]:
	last_error: Optional[Exception] = None
	for attempt in range(1, retry + 1):
		try:
			# Use Chat Completions with JSON object enforcement
			resp = client.chat.completions.create(
				model=model,
				messages=[
					{"role": "system", "content": "You are a quantitative FX analyst. Output must be a single JSON object with only the top-level key 'predictions' containing horizon keys '1m', '3m', '5m', '10m'. Do not include any other keys or echo inputs."},
					{"role": "user", "content": prompt_text},
				],
				response_format={"type": "json_object"},
				temperature=0.0,
			)
			content_text = resp.choices[0].message.content if resp and resp.choices else None  # type: ignore
			if not content_text:
				raise RuntimeError("Empty response content from model")
			parsed = json.loads(content_text)
			if not isinstance(parsed, dict):
				raise ValueError("Parsed response is not a JSON object")
			return parsed, content_text
		except Exception as e:  # pragma: no cover
			last_error = e
			time.sleep(sleep_between_sec)
			continue
	assert last_error is not None
	raise last_error


def extract_predictions(parsed: Dict[str, Any]) -> RowPrediction:
	preds = parsed.get("predictions") if isinstance(parsed, dict) else None

	def _get(h: str) -> Optional[ModelPrediction]:
		if not preds or h not in preds:
			return None
		obj = preds[h]
		try:
			return ModelPrediction(
				will_go_up=bool(obj["will_go_up"]),
				probability_up=float(obj["probability_up"]),
				expected_return_pct=float(obj["expected_return_pct"]),
				rationale=str(obj.get("rationale", ""))[:500],
			)
		except Exception:
			return None

	return RowPrediction(
		pred_h1m=_get("1m"),
		pred_h3m=_get("3m"),
		pred_h5m=_get("5m"),
		pred_h10m=_get("10m"),
	)


def rate_limit_sleep(last_request_time: List[float], rpm: float) -> None:
	if rpm <= 0:
		return
	interval = 60.0 / rpm
	now = time.time()
	if last_request_time[0] == 0.0:
		last_request_time[0] = now
		return
	elapsed = now - last_request_time[0]
	if elapsed < interval:
		time.sleep(interval - elapsed)
	last_request_time[0] = time.time()


def compute_metrics(df: pd.DataFrame) -> Dict[str, Any]:
	results: Dict[str, Any] = {}
	for h in HORIZONS_MIN:
		label_col = f"realized_return_pct_{h}m"
		pred_up_col = f"pred_{h}m_will_go_up"
		prob_col = f"pred_{h}m_probability_up"
		exp_col = f"pred_{h}m_expected_return_pct"

		valid_mask = df[label_col].notna() & df[pred_up_col].notna()
		if valid_mask.sum() == 0:
			results[f"acc_{h}m"] = None
			results[f"mae_{h}m"] = None
			results[f"rmse_{h}m"] = None
			results[f"r2_{h}m"] = None
			continue

		labels = df.loc[valid_mask, label_col].values
		pred_up = df.loc[valid_mask, pred_up_col].astype(bool).values
		direction_truth = labels > 0
		acc = float((pred_up == direction_truth).mean())

		exp_vals = df.loc[valid_mask, exp_col].astype(float).values
		mae = float(np.mean(np.abs(exp_vals - labels)))
		rmse = float(np.sqrt(np.mean((exp_vals - labels) ** 2)))

		# Simple R^2
		var_y = float(np.var(labels))
		ss_res = float(np.sum((labels - exp_vals) ** 2))
		ss_tot = float(np.sum((labels - np.mean(labels)) ** 2))
		r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else float("nan")

		results[f"acc_{h}m"] = acc
		results[f"mae_{h}m"] = mae
		results[f"rmse_{h}m"] = rmse
		results[f"r2_{h}m"] = r2
	return results


def main() -> None:
	cwd = Path(os.path.abspath(os.path.dirname(__file__)))
	# Load .env before parsing args so defaults can be read from env
	dotenv_path = cwd / ".env"
	loaded_env = load_env_file(dotenv_path)

	args = parse_args()
	# Prepare timestamped output directory and paths
	ts = time.strftime("%Y%m%d_%H%M%S")
	out_dir = Path(args.out_dir)
	if not out_dir.is_absolute():
		out_dir = (cwd / out_dir).resolve()
	out_dir.mkdir(parents=True, exist_ok=True)

	# Determine output file path
	if args.out == "results.csv" or not args.out:
		out_path = out_dir / f"results_{ts}.csv"
	else:
		p = Path(args.out)
		out_path = p if p.is_absolute() else (cwd / p).resolve()

	# Determine cache file path
	if args.cache == "cache.jsonl" or not args.cache:
		cache_path = out_dir / f"cache_{ts}.jsonl"
	else:
		p = Path(args.cache)
		cache_path = p if p.is_absolute() else (cwd / p).resolve()

	# Determine prompts file path
	if args.prompts == "prompts.jsonl" or not args.prompts:
		prompts_path = out_dir / f"prompts_{ts}.jsonl"
	else:
		p = Path(args.prompts)
		prompts_path = p if p.is_absolute() else (cwd / p).resolve()

	# Determine responses file path
	if args.responses == "responses.jsonl" or not args.responses:
		responses_path = out_dir / f"responses_{ts}.jsonl"
	else:
		p = Path(args.responses)
		responses_path = p if p.is_absolute() else (cwd / p).resolve()

	log_path = cwd / args.log
	configure_logging(log_path)
	logger = logging.getLogger(__name__)

	if loaded_env:
		logger.info("Loaded environment from %s", str(dotenv_path))
	else:
		logger.info("No .env file found at %s (skipping)", str(dotenv_path))

	# Resolve default model from env if not provided
	if args.model is None:
		args.model = os.environ.get("OPENAI_MODEL") or os.environ.get("MODEL") or "gpt-4o-mini"
		logger.info("Using model: %s", args.model)

	logger.info("Output directory: %s", str(out_dir))
	logger.info("Results path: %s", str(out_path))
	logger.info("Cache path: %s", str(cache_path))
	logger.info("Prompts path: %s", str(prompts_path))
	logger.info("Responses path: %s", str(responses_path))

	logger.info("Loading dataset: %s", args.dataset)
	dataset_path = Path(args.dataset)
	if not dataset_path.is_absolute():
		dataset_path = (cwd / dataset_path).resolve()
	if not dataset_path.exists():
		logger.error("Dataset not found: %s", str(dataset_path))
		sys.exit(1)

	df_raw = load_dataset(dataset_path)
	logger.info("Loaded rows: %d", len(df_raw))

	logger.info("Computing realized returns")
	df = compute_realized_returns(df_raw)

	# Subsample controls
	stride = max(1, int(args.stride))
	indices = list(range(0, len(df), stride))
	if args.max_rows is not None:
		indices = indices[: int(args.max_rows)]
	logger.info("Processing rows: %d (from %d total)", len(indices), len(df))

	# Prepare model client
	if not _HAS_OPENAI:
		logger.error("openai package not available. Please install dependencies in venv.")
		sys.exit(2)
	api_key = os.environ.get("OPENAI_API_KEY")
	if not api_key:
		logger.error("OPENAI_API_KEY not set in environment")
		sys.exit(3)

	# Support base URL / org / project from env if present
	base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")
	organization = os.environ.get("OPENAI_ORG_ID") or os.environ.get("OPENAI_ORGANIZATION")
	project = os.environ.get("OPENAI_PROJECT")

	# Ensure proxies are not passed to the HTTP client to avoid compatibility issues
	for k in [
		"HTTP_PROXY",
		"HTTPS_PROXY",
		"ALL_PROXY",
		"http_proxy",
		"https_proxy",
		"all_proxy",
	]:
		if k in os.environ:
			os.environ.pop(k, None)

	try:
		client = OpenAI(api_key=api_key, base_url=base_url, organization=organization, project=project)
	except TypeError as e:
		# Fallback retry without optional params if the installed SDK doesn't support them
		logger.warning("Retrying OpenAI client init due to error: %s", e)
		client = OpenAI(api_key=api_key)

	cache = JsonlCache(cache_path)

	# Output CSV writer and prompts/responses writers
	fieldnames = [
		"datetime",
		"close",
		# realized
		"realized_return_pct_1m",
		"realized_return_pct_3m",
		"realized_return_pct_5m",
		"realized_return_pct_10m",
		# final closes
		"final_close_1m",
		"final_close_3m",
		"final_close_5m",
		"final_close_10m",
		# predictions
		"pred_1m_will_go_up",
		"pred_1m_probability_up",
		"pred_1m_expected_return_pct",
		"pred_1m_rationale",
		"pred_3m_will_go_up",
		"pred_3m_probability_up",
		"pred_3m_expected_return_pct",
		"pred_3m_rationale",
		"pred_5m_will_go_up",
		"pred_5m_probability_up",
		"pred_5m_expected_return_pct",
		"pred_5m_rationale",
		"pred_10m_will_go_up",
		"pred_10m_probability_up",
		"pred_10m_expected_return_pct",
		"pred_10m_rationale",
	]

	if out_path.exists():
		logger.info("Output file exists, it will be overwritten: %s", str(out_path))

	last_request_time = [0.0]

	with out_path.open("w", newline="", encoding="utf-8") as csvfile, prompts_path.open("w", encoding="utf-8") as promptfile, responses_path.open("w", encoding="utf-8") as responsefile:
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		writer.writeheader()

		invalid_extracts = 0

		for idx in tqdm(indices, desc="Processing", unit="row"):
			row = df.iloc[idx]

			# Build prompt
			trailing_context: Optional[List[Dict[str, Any]]] = None
			if args.window and args.window > 0:
				if idx < args.window:
					logging.warning("Row %d has no trailing context (window=%d)", idx, args.window)
					continue
				start_idx = max(0, idx - args.window)
				ctx_df = df_raw.iloc[start_idx: idx]
				trailing_context = [ctx_df.iloc[i].to_dict() for i in range(len(ctx_df))]

			current_snapshot = df_raw.iloc[idx].to_dict()
			prompt_text = build_prompt(current_snapshot, trailing_context)

			# Save prompt
			cache_key = hashlib.sha256(json.dumps(current_snapshot, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
			prompt_line = {
				"row_index": idx,
				"datetime": current_snapshot.get("datetime"),
				"cache_key": cache_key,
				"prompt": prompt_text,
			}
			promptfile.write(json.dumps(prompt_line, ensure_ascii=False) + "\n")

			# Cache lookup / model call
			cached = cache.get(cache_key)
			if cached is None:
				# Rate limit
				rate_limit_sleep(last_request_time, float(args.rpm))
				parsed, raw_text = call_model(
					client=client,
					model=args.model,
					prompt_text=prompt_text,
					timeout_sec=int(args.timeout),
					retry=3,
					sleep_between_sec=1.5,
				)
				cache.set(cache_key, parsed)
				response_obj = parsed
				# Log truncated raw response to console
				log_snippet = (raw_text or "")[:400].replace("\n", " ")
				logging.info("Raw model response (row %d) [trunc]: %s", idx, log_snippet)
				# Save full response entry
				responsefile.write(json.dumps({
					"row_index": idx,
					"datetime": current_snapshot.get("datetime"),
					"cache_key": cache_key,
					"from_cache": False,
					"raw_text": raw_text,
					"parsed_top_level_keys": list(response_obj.keys()) if isinstance(response_obj, dict) else None,
				}, ensure_ascii=False) + "\n")
			else:
				response_obj = cached
				responsefile.write(json.dumps({
					"row_index": idx,
					"datetime": current_snapshot.get("datetime"),
					"cache_key": cache_key,
					"from_cache": True,
					"raw_text": None,
					"parsed_top_level_keys": list(response_obj.keys()) if isinstance(response_obj, dict) else None,
				}, ensure_ascii=False) + "\n")

			row_pred = extract_predictions(response_obj)

			# Validate extraction
			missing = []
			if not isinstance(response_obj, dict) or "predictions" not in response_obj:
				missing.append("predictions")
			else:
				for h in ["1m", "3m", "5m", "10m"]:
					if h not in response_obj["predictions"]:
						missing.append(h)
			if missing:
				invalid_extracts += 1
				logging.warning("Row %d missing sections in model JSON: %s", idx, ", ".join(missing))

			write_row: Dict[str, Any] = {
				"datetime": current_snapshot.get("datetime"),
				"close": current_snapshot.get("close"),
				"realized_return_pct_1m": row.get("realized_return_pct_1m"),
				"realized_return_pct_3m": row.get("realized_return_pct_3m"),
				"realized_return_pct_5m": row.get("realized_return_pct_5m"),
				"realized_return_pct_10m": row.get("realized_return_pct_10m"),
				"final_close_1m": row.get("final_close_1m"),
				"final_close_3m": row.get("final_close_3m"),
				"final_close_5m": row.get("final_close_5m"),
				"final_close_10m": row.get("final_close_10m"),
			}
			write_row.update(row_pred.to_row_dict())
			writer.writerow(write_row)

		if invalid_extracts:
			logger.info("Rows with invalid/missing prediction JSON: %d", invalid_extracts)

	logger.info("Loading results for metrics: %s", str(out_path))
	df_out = pd.read_csv(out_path)

	metrics = compute_metrics(df_out)
	logger.info("Final metrics:")
	for k in sorted(metrics.keys()):
		logger.info("%s = %s", k, metrics[k])

	logger.info("Done. Output written to: %s", str(out_path))
	logger.info("Cache written to: %s", str(cache_path))
	logger.info("Prompts written to: %s", str(prompts_path))
	logger.info("Responses written to: %s", str(responses_path))
	logger.info("Log saved to: %s", str(log_path))


if __name__ == "__main__":
	main()
