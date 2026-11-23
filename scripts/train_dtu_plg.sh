#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
: "${DTU_ROOT:?请设置 DTU_ROOT 为 DTU 训练数据目录}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
exec "${PYTHON:-python}" train_dtu.py --mode train --dataset dtu_yao_vit   --trainpath "$DTU_ROOT" --trainlist lists/dtu/train.txt --testlist lists/dtu/test.txt   --logdir "${LOG_DIR:-checkpoints/plgmvs_dtu}" --batch_size "${BATCH_SIZE:-4}"   --epochs 15 --lr 0.001 --wd 0.0001 --lrepochs 10,12,14:2   --ndepths 8,8,4,4 --attn_temp 2 --summary_freq 100 "$@"
