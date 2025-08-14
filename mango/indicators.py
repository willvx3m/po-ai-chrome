#!/usr/bin/env python3

import numpy as np
import pandas as pd


def calculate_wilder_rsi(close_series: pd.Series, period: int = 7) -> pd.Series:
	# Price changes
	delta = close_series.diff()
	gain = delta.clip(lower=0)
	loss = -delta.clip(upper=0)

	# Wilder's smoothing via EMA with alpha = 1/period
	avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
	avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()

	rs = avg_gain / (avg_loss + 1e-12)
	rsi = 100 - (100 / (1 + rs))
	return rsi


def clamp_round(series: pd.Series, lo: float, hi: float, decimals: int = 1) -> pd.Series:
	return series.clip(lower=lo, upper=hi).round(decimals) 