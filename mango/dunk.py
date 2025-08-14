#!/usr/bin/env python3

# This script converts EURUSD candles to snapshot features.
# Snapshot vector: [datetime, close, ema_fast_change_pct_3m, ema_slow_change_pct_6m, ema_spread_pct, price_to_ema_fast_pct, vol_std_10m_pct, macd_hist_pct, rsi14_centered, ret_log_5m_pct, ret_log_15m_pct]
#  - ema_fast_change_pct_3m: percentage change of EMA(6) over 3 minutes
#  - ema_slow_change_pct_6m: percentage change of EMA(13) over 6 minutes
#  - ema_spread_pct: percentage EMA(6)-EMA(13) spread relative to EMA(13)
#  - price_to_ema_fast_pct: percentage distance of close to EMA(6)
#  - vol_std_10m_pct: 100 * std of 1m log-returns over 10 minutes (then scaled x10)
#  - macd_hist_pct: 100 * (MACD(6,13) - Signal(4)) / EMA(13) (then scaled x10)
#  - rsi14_centered: (RSI(7) - 50) / 50 in [-1, 1]
#  - ret_log_5m_pct: 100 * ln(close/close[-5]) (then scaled x10)
#  - ret_log_15m_pct: 100 * ln(close/close[-15]) (then scaled x10)
# It assumes a 1-minute timeframe.
# It outputs a JSON file with the features. Logging goes to console and mango/log.txt.

import argparse
import json
import logging
import os
from typing import List

import numpy as np
import pandas as pd

try:
	from .log_utils import configure_logging
	from .indicators import calculate_wilder_rsi, clamp_round
except Exception:
	# Fallback for direct script execution
	import sys as _sys
	_script_dir = os.path.abspath(os.path.dirname(__file__))
	if _script_dir not in _sys.path:
		_sys.path.append(_script_dir)
	from log_utils import configure_logging  # type: ignore
	from indicators import calculate_wilder_rsi, clamp_round  # type: ignore


def compute_snapshots(df: pd.DataFrame) -> pd.DataFrame:
	# Ensure required fields exist
	required_fields: List[str] = ["close", "datetime_point"]
	missing = [f for f in required_fields if f not in df.columns]
	if missing:
		raise ValueError(f"Missing required fields in input JSON: {missing}")

	close = df["close"]

	# Config (short-term)
	ema_fast_span = 6
	ema_slow_span = 13
	ema_fast_lookback_m = 3
	ema_slow_lookback_m = 6
	rsi_period = 7
	macd_signal_span = 4
	vol_window_m = 10
	logging.info(
		f"Config: EMA fast={ema_fast_span}, slow={ema_slow_span}; lookbacks: fast {ema_fast_lookback_m}m, slow {ema_slow_lookback_m}m; RSI={rsi_period}; MACD signal={macd_signal_span}; vol window={vol_window_m}m; scaling x10 on percent/log features"
	)

	# Compute EMAs
	df["ema_fast"] = close.ewm(span=ema_fast_span, adjust=False).mean()
	df["ema_slow"] = close.ewm(span=ema_slow_span, adjust=False).mean()

	# Compute RSI (Wilder)
	df["rsi"] = calculate_wilder_rsi(close, period=rsi_period)

	# Lookbacks (minutes)
	ema_fast_prev = df["ema_fast"].shift(ema_fast_lookback_m)
	ema_slow_prev = df["ema_slow"].shift(ema_slow_lookback_m)

	# Percent changes for EMAs (percent units first)
	ema_fast_change_pct_3m = 100.0 * (df["ema_fast"] - ema_fast_prev) / (ema_fast_prev.replace(0, np.nan))
	ema_slow_change_pct_6m = 100.0 * (df["ema_slow"] - ema_slow_prev) / (ema_slow_prev.replace(0, np.nan))

	# Additional features (percent units first)
	ema_spread_pct = 100.0 * (df["ema_fast"] - df["ema_slow"]) / (df["ema_slow"].replace(0, np.nan))
	price_to_ema_fast_pct = 100.0 * (close - df["ema_fast"]) / (df["ema_fast"].replace(0, np.nan))

	# Volatility: std of 1m log returns over vol_window_m
	log_ret_1m = np.log(close / close.shift(1))
	vol_std_10m_pct = 100.0 * log_ret_1m.rolling(window=vol_window_m).std()

	# MACD histogram percent based on fast/slow
	macd = df["ema_fast"] - df["ema_slow"]
	signal = macd.ewm(span=macd_signal_span, adjust=False).mean()
	macd_hist_pct = 100.0 * (macd - signal) / (df["ema_slow"].replace(0, np.nan))

	# Centered RSI
	rsi14_centered = (df["rsi"] - 50.0) / 50.0

	# Log returns horizons (percent units first)
	ret_log_5m_pct = 100.0 * np.log(close / close.shift(5))
	ret_log_15m_pct = 100.0 * np.log(close / close.shift(15))

	# Scale percent/log-derived features by 10 to enhance resolution, then clamp and round to 1 decimal
	ema_fast_change_pct_3m = clamp_round(10.0 * ema_fast_change_pct_3m, -10.0, 10.0)
	ema_slow_change_pct_6m = clamp_round(10.0 * ema_slow_change_pct_6m, -10.0, 10.0)
	ema_spread_pct = clamp_round(10.0 * ema_spread_pct, -10.0, 10.0)
	price_to_ema_fast_pct = clamp_round(10.0 * price_to_ema_fast_pct, -10.0, 10.0)
	vol_std_10m_pct = clamp_round(10.0 * vol_std_10m_pct, 0.0, 10.0)
	macd_hist_pct = clamp_round(10.0 * macd_hist_pct, -10.0, 10.0)
	rsi14_centered = clamp_round(rsi14_centered, -1.0, 1.0)
	ret_log_5m_pct = clamp_round(10.0 * ret_log_5m_pct, -10.0, 10.0)
	ret_log_15m_pct = clamp_round(10.0 * ret_log_15m_pct, -10.0, 10.0)

	# Assemble output
	out = pd.DataFrame({
		"datetime": df["datetime_point"],
		"close": close,
		"ema_fast_change_pct_3m": ema_fast_change_pct_3m,
		"ema_slow_change_pct_6m": ema_slow_change_pct_6m,
		"ema_spread_pct": ema_spread_pct,
		"price_to_ema_fast_pct": price_to_ema_fast_pct,
		"vol_std_10m_pct": vol_std_10m_pct,
		"macd_hist_pct": macd_hist_pct,
		"rsi14_centered": rsi14_centered,
		"ret_log_5m_pct": ret_log_5m_pct,
		"ret_log_15m_pct": ret_log_15m_pct,
	})

	# Keep warmup window at 38 (per user request)
	min_required_history = 38
	if len(out) > 0:
		logging.info(f"Enforcing minimum history of {min_required_history} candles; dropping first {min_required_history} rows")
		out = out.iloc[min_required_history:].reset_index(drop=True)

	# Drop rows with NaNs introduced by lookbacks/rolling windows
	needed = [
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
	num_before = len(out)
	num_missing = out[needed].isna().any(axis=1).sum()
	logging.info(f"Snapshot rows before drop: {num_before}, dropping {num_missing} due to warmup/NaNs")
	out = out.dropna(subset=needed).reset_index(drop=True)

	# Ranges logging
	def log_range(name: str, series: pd.Series) -> None:
		logging.info(f"{name} range: [{series.min()}, {series.max()}]")

	log_range("ema_fast_change_pct_3m", out["ema_fast_change_pct_3m"])
	log_range("ema_slow_change_pct_6m", out["ema_slow_change_pct_6m"])
	log_range("ema_spread_pct", out["ema_spread_pct"])
	log_range("price_to_ema_fast_pct", out["price_to_ema_fast_pct"])
	log_range("vol_std_10m_pct", out["vol_std_10m_pct"])
	log_range("macd_hist_pct", out["macd_hist_pct"])
	log_range("rsi14_centered", out["rsi14_centered"])
	log_range("ret_log_5m_pct", out["ret_log_5m_pct"])
	log_range("ret_log_15m_pct", out["ret_log_15m_pct"])
	log_range("close", out["close"])

	return out


def main() -> None:
	configure_logging(None, "log-dunk.txt")

	parser = argparse.ArgumentParser(description="Convert EURUSD candles to snapshot features.")
	parser.add_argument(
		"--input",
		type=str,
		default=os.path.join(os.path.dirname(__file__), "eurusd.json"),
		help="Path to input eurusd.json (JSON array of candle objects)",
	)
	parser.add_argument(
		"--output",
		type=str,
		default=os.path.join(os.path.dirname(__file__), "eurusd-snapshots.json"),
		help="Path to output JSON file",
	)
	args = parser.parse_args()

	logging.info(f"Reading input: {args.input}")
	with open(args.input, "r", encoding="utf-8") as f:
		data = json.load(f)

	# Log detected fields
	if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
		logging.info(f"Detected input fields: {sorted(list(data[0].keys()))}")
	logging.info("Assuming timeframe: 1 minute (short-term config)")

	df = pd.DataFrame(data)
	logging.info(f"Loaded {len(df)} candles")

	# Compute snapshots
	logging.info("Computing indicators and snapshots...")
	out_df = compute_snapshots(df)
	logging.info(f"Generated {len(out_df)} snapshots")

	# Datetime coverage logging
	if len(out_df) > 0:
		logging.info(f"Snapshot datetime range: {out_df['datetime'].iloc[0]} -> {out_df['datetime'].iloc[-1]}")

	# Write output
	records = out_df.to_dict(orient="records")
	with open(args.output, "w", encoding="utf-8") as f:
		json.dump(records, f, ensure_ascii=False)
	logging.info(f"Wrote snapshots JSON: {args.output}")


if __name__ == "__main__":
	main()
