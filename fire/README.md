## Fire Backtesting Runner

### Overview
Runs an LLM-driven backtest over EURUSD snapshot vectors and saves predictions, realized returns, prompts, and responses.

- Input dataset: `dataset/eurusd-snapshots.json` (JSON array of arrays or array of dicts)
- Model queried per snapshot for horizons 1m, 3m, 5m, 10m
- Outputs saved under timestamped files in `fire/results/`
- Logs also written to `fire/log.txt` in addition to console [[memory:4776439]]

### Snapshot vector format
Each row is either a list (exact order) or a dict with the same keys:
1. datetime
2. close
3. ema_fast_change_pct_3m
4. ema_slow_change_pct_6m
5. ema_spread_pct
6. price_to_ema_fast_pct
7. vol_std_10m_pct
8. macd_hist_pct
9. rsi14_centered
10. ret_log_5m_pct
11. ret_log_15m_pct

### Setup
```bash
cd fire
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install openai==1.40.0 pandas==2.2.2 numpy==1.26.4 tqdm==4.66.4 python-dateutil==2.9.0.post0 httpx==0.27.2
```

Create `.env` in `fire/`:
```bash
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
# Optional:
# OPENAI_BASE_URL=...
# OPENAI_ORGANIZATION=...
# OPENAI_PROJECT=...
```

### Run
Basic run (cost-controlled):
```bash
source .venv/bin/activate
python run.py --dataset ../dataset/eurusd-snapshots.json \
  --max-rows 2000 --stride 10 --rpm 60 --timeout 40
```

Full run (expensive for 100k rows):
```bash
python run.py --rpm 60 --timeout 40
```

Default behavior:
- `--window 10` uses last 10 minutes as trailing context
- Uses `OPENAI_MODEL` from `.env` (falls back to `gpt-4o-mini` if not set)
- Rate limits requests (default `--rpm 200`); cache prevents re-calls

### Outputs
- Directory: `fire/results/`
  - `results_YYYYMMDD_HHMMSS.csv`: per-row outputs
    - Columns include:
      - `datetime`, `close`
      - Realized returns: `realized_return_pct_1m`, `realized_return_pct_3m`, `realized_return_pct_5m`, `realized_return_pct_10m`
        - Computed as `100 * ln(close[t+h] / close[t])`
      - Final close prices: `final_close_1m`, `final_close_3m`, `final_close_5m`, `final_close_10m`
        - These are `close[t+h]` aligned to the realized returns
      - Predictions per horizon: `pred_{h}m_will_go_up`, `pred_{h}m_probability_up`, `pred_{h}m_expected_return_pct`, `pred_{h}m_rationale`
  - `cache_YYYYMMDD_HHMMSS.jsonl`: LLM response cache keyed by snapshot hash (used for resume-safe runs)
  - `prompts_YYYYMMDD_HHMMSS.jsonl`: full prompts per row
  - `responses_YYYYMMDD_HHMMSS.jsonl`: raw model responses and parsed-key summary per row
- Logs: `fire/log.txt` [[memory:4776439]]

### CLI options
```text
--dataset        Path to dataset JSON (default ../dataset/eurusd-snapshots.json)
--model          OpenAI model name (default from .env OPENAI_MODEL or gpt-4o-mini)
--window         Trailing context size in minutes (default 10; 0 = no context)
--max-rows       Limit number of rows to process
--stride         Process every Nth row (cost control)
--out-dir        Directory for timestamped outputs (default results)
--cache          Custom cache file path (otherwise timestamped under results)
--rpm            Requests per minute rate limit (default 200)
--timeout        Request timeout seconds (default 30)
--out            Custom CSV path (otherwise timestamped under results)
--prompts        Custom prompts JSONL path
--responses      Custom responses JSONL path
--log            Log file path (default log.txt in fire)
--p-threshold    Probability threshold for optional strategy (not used in CSV metrics)
--tx-cost        Transaction cost per trade (return pct units; default 0)
```

### Debugging & troubleshooting
- If predictions aren’t extracted, see `fire/log.txt` for warnings about missing JSON sections and check `results/responses_*.jsonl` raw entries.
- If the SDK errors with proxies, the script unsets proxy variables before init; ensure `.env` does not set incompatible proxy keys.
- For rate limits, increase `--rpm` only if your quota allows; use `--stride` and `--max-rows` to control cost.

### Notes
- The cache file allows stopping and resuming runs without re-calling the model.
- Realized returns are based on log returns in percent; for small moves this approximates simple percent change. 