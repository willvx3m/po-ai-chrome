# install pytorch at venv-build
# start from downloading pytorch source code
# use develop branch to build
# use 6 threads to build
# use cuda 12.0 to build
# compataible CC version is 12.0
bash -lc 'set -euo pipefail; LOG=/mnt/WORK/Project/MegaVX/po-ai/mango/build-pytorch.log; rm -f "$LOG"; echo Start: $(date) | tee -a "$LOG"; cd /tmp; rm -rf pytorch; git clone --recursive https://github.com/pytorch/pytorch.git >>"$LOG" 2>&1; cd pytorch; # optional: checkout a tag; 
source /mnt/WORK/Project/MegaVX/po-ai/mango/venv-build/bin/activate; pip install -r requirements.txt >>"$LOG" 2>&1; export CUDA_HOME=/usr/local/cuda; export USE_CUDA=1; export USE_CUDNN=1; export TORCH_CUDA_ARCH_LIST="12.0"; export MAX_JOBS=6; export PYTORCH_NVCC_FLAGS="-Xfatbin -compress-all"; python setup.py develop >>"$LOG" 2>&1; echo Done: $(date) | tee -a "$LOG"'