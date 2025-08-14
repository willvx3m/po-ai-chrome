# Snapshot Preprocessing (EURUSD)

This directory contains a preprocessing tool that converts 1-minute EURUSD candles into per-minute snapshot vectors for short-term modeling.

## Script
- `dunk.py`: main CLI tool to generate snapshots.
- `indicators.py`: RSI and small utilities.
- `log_utils.py`: logging configuration (console + `mango/log.txt`).

## Input
- File: `eurusd.json` (JSON array of candle objects) with at least:
  - `datetime_point` (e.g., "2025-05-09 12:45")
  - `open`, `high`, `low`, `close`
- Data is assumed sorted by minute, no gaps.

## Output
- File: `eurusd-snapshots.json` (JSON array of snapshot records)
- One snapshot per minute after warmup.

## Snapshot vector (fields)
- `datetime`: copy of `datetime_point`
- `close`: last price (precision unchanged)
- `ema_fast_change_pct_3m`: 100 * (EMA(6)[t] − EMA(6)[t−3]) / EMA(6)[t−3], scaled ×10, clamped to [-10.0, 10.0], 1 decimal
- `ema_slow_change_pct_6m`: 100 * (EMA(13)[t] − EMA(13)[t−6]) / EMA(13)[t−6], scaled ×10, clamped to [-10.0, 10.0], 1 decimal
- `ema_spread_pct`: 100 * (EMA(6) − EMA(13)) / EMA(13), scaled ×10, clamped to [-10.0, 10.0], 1 decimal
- `price_to_ema_fast_pct`: 100 * (close − EMA(6)) / EMA(6), scaled ×10, clamped to [-10.0, 10.0], 1 decimal
- `vol_std_10m_pct`: 100 * std(1m log-returns over last 10), scaled ×10, clamped to [0.0, 10.0], 1 decimal
- `macd_hist_pct`: 100 * (MACD(6,13) − Signal(4)) / EMA(13), scaled ×10, clamped to [-10.0, 10.0], 1 decimal
- `rsi14_centered`: (RSI(7) − 50) / 50, clamped to [-1.0, 1.0], 1 decimal
- `ret_log_5m_pct`: 100 * ln(close[t]/close[t−5]), scaled ×10, clamped to [-10.0, 10.0], 1 decimal
- `ret_log_15m_pct`: 100 * ln(close[t]/close[t−15]), scaled ×10, clamped to [-10.0, 10.0], 1 decimal

Notes:
- “scaled ×10” means the computed percent/log value is multiplied by 10 before clamping/rounding to preserve short-term resolution.
- All clamped/rounded fields use 1 decimal.

## Indicator configuration (short-term)
- EMA fast/slow: 6 / 13
- Lookbacks: 3m (fast), 6m (slow)
- RSI (Wilder): 7
- MACD signal: 4
- Volatility window: 10 minutes (std of 1m log-returns)

## Warmup
- First 38 candles are skipped to ensure all features have sufficient history.
- After warmup, all snapshots are valid (no NaNs).

## Logging
- Preprocessing (`dunk.py`): logs to console and `mango/log-dunk.txt`.
- Training/Prediction: logs to console and `mango/log.txt`.
- Logs include detected fields, config, warmup drops, feature ranges, splits, metrics, and calibration output paths.

## Usage: Generate snapshots
Default paths (input: `mango/eurusd.json`, output: `mango/eurusd-snapshots.json`):

```bash
python3 mango/dunk.py
```

Custom paths:
```bash
python3 mango/dunk.py \
  --input /path/to/eurusd.json \
  --output /path/to/eurusd-snapshots.json
```

## Model: Multi-horizon classifier (PyTorch)
- Single multi-task MLP (shared trunk) with 3 heads (3 classes each: -1/0/1 mapped to indices 0/1/2).
- Labels per horizon h ∈ {3,5,10} minutes using pip threshold:
  - delta = close[t+h] − close[t]
  - if delta > (pip_value × pip_threshold_pips) → class 2 (up)
  - elif delta < −(pip_value × pip_threshold_pips) → class 0 (down)
  - else → class 1 (not sure)
- Default: pip_value=0.0001, pip_threshold_pips=1.
- Time-based split: 70% train, 15% val, 15% test.
- Class weights computed from training set per horizon.
- Calibration plots saved as PNGs per horizon.

### Train
From project root (recommended):
```bash
# CPU or GPU (auto-detects; falls back to CPU if GPU kernels unsupported)
python3 -m mango.train \
  --input mango/eurusd-snapshots.json \
  --pip-value 0.0001 \
  --pip-threshold-pips 1 \
  --epochs 30 \
  --batch-size 8192
```
Artifacts written to `mango/checkpoints/`: `model.pt`, `features.json`, `config.json`, `calibration_*.png`.

### Predict (single snapshot)
```bash
python3 -m mango.predict --input '{"close":1.1, "ema_fast_change_pct_3m":0.2, "ema_slow_change_pct_6m":0.1, "ema_spread_pct":0.0, "price_to_ema_fast_pct":-0.1, "vol_std_10m_pct":0.2, "macd_hist_pct":0.0, "rsi14_centered":-0.2, "ret_log_5m_pct":0.1, "ret_log_15m_pct":-0.1}'
```
Output JSON includes per-horizon probability vectors (3 classes) and predicted class (-1/0/1).

## Environment
Use a local venv (example under `mango/venv`):
```bash
python3 -m venv mango/venv
mango/venv/bin/pip install -U pip numpy pandas matplotlib torch
```

### GPU
- Training auto-detects CUDA; if kernels are not compatible with your GPU arch, it logs a warning and falls back to CPU.
- To use GPU, install a CUDA-enabled PyTorch wheel matching your driver/arch (see PyTorch selector). If your GPU is newer and unsupported, wait for a compatible build or use CPU in the meantime.

## Build PyTorch from source (optional)
If your GPU arch isn’t supported by current wheels, you can build PyTorch from source targeting your SM version.

- Prereqs: NVIDIA driver, CUDA Toolkit 12.x (e.g., `/usr/local/cuda`), cuDNN, git, cmake, ninja, python3-dev.
- Use your venv and install Python deps:
```bash
source mango/venv/bin/activate
pip uninstall -y torch || true
pip install -U pip setuptools wheel typing-extensions numpy jinja2 fsspec sympy filelock networkx
```
- Clone and build:
```bash
cd /tmp
git clone --recursive https://github.com/pytorch/pytorch.git
cd pytorch
export CUDA_HOME=/usr/local/cuda
export USE_CUDA=1 USE_CUDNN=1 TORCH_CUDA_ARCH_LIST="12.0" MAX_JOBS=$(nproc)
python setup.py develop
```
- Verify:
```bash
python - <<'PY'
import torch
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
PY
```

## Try PyTorch nightly (optional)
Nightly wheels may add support for newer GPUs earlier than stable.

- Create a separate venv:
```bash
python3 -m venv mango/venv-nightly
mango/venv-nightly/bin/pip install -U pip
# Try a CUDA nightly (adjust cu version per PyTorch selector)
mango/venv-nightly/bin/pip install --pre --upgrade torch --index-url https://download.pytorch.org/whl/nightly/cu128
pip install "numpy<2.0"
```
- Verify GPU:
```bash
mango/venv-nightly/bin/python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```
If unsupported, fall back to CPU or build from source.

## Example output record
```json
{
  "datetime": "2025-05-09 13:10",
  "close": 1.1282,
  "ema_fast_change_pct_3m": -0.1,
  "ema_slow_change_pct_6m": -0.1,
  "ema_spread_pct": -0.0,
  "price_to_ema_fast_pct": -0.1,
  "vol_std_10m_pct": 0.1,
  "macd_hist_pct": -0.0,
  "rsi14_centered": -0.2,
  "ret_log_5m_pct": 0.1,
  "ret_log_15m_pct": -0.1
}
```

---

## venv-build (preferred)
Use the prebuilt environment under `mango/venv-build`.

```bash
# Install basic deps if missing
/mnt/WORK/Project/MegaVX/po-ai/mango/venv-build/bin/pip install -U pip numpy pandas matplotlib
```

## Training presets and options
- `--model {small,base,xl}`: preset sizes (base is default).
- Overrides: `--hidden-dim`, `--depth`, `--dropout`.
- Throughput: `--batch-size {int|auto}` (auto searches to ~80% VRAM), `--grad-accum-steps N`.
- Performance: `--allow-tf32` (Ampere+), `--compile` (PyTorch 2.0+, off by default).

## Strongest run (16GB GPU)
```bash
# GPU monitor (optional)
nohup nvidia-smi --query-gpu=timestamp,name,memory.total,memory.used,utilization.gpu --format=csv -l 5 \
  >> /mnt/WORK/Project/MegaVX/po-ai/mango/gpu_monitor.log 2>&1 &

# Train with XL preset, auto batch to ~80% VRAM, TF32, 10 epochs, grad accumulation, optional compile
/mnt/WORK/Project/MegaVX/po-ai/mango/venv-build/bin/python \
  /mnt/WORK/Project/MegaVX/po-ai/mango/train.py \
  --input /mnt/WORK/Project/MegaVX/po-ai/mango/eurusd-snapshots.json \
  --model xl \
  --batch-size auto \
  --allow-tf32 \
  --epochs 10 \
  --grad-accum-steps 2 \
  --compile
```

View logs:
```bash
# Training logs
tail -f /mnt/WORK/Project/MegaVX/po-ai/mango/log.txt
# GPU monitor
tail -f /mnt/WORK/Project/MegaVX/po-ai/mango/gpu_monitor.log
``` 