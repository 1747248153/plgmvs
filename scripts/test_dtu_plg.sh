#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
: "${DTU_TEST_ROOT:?请设置 DTU_TEST_ROOT 为 DTU 测试数据目录}"
: "${PLG_CHECKPOINT:?请设置 PLG_CHECKPOINT 为完整 PLG checkpoint 路径}"
exec "${PYTHON:-python}" test_dtu_dypcd.py --dataset general_eval_vit --batch_size 1   --testpath "$DTU_TEST_ROOT" --testlist lists/dtu/test.txt --loadckpt "$PLG_CHECKPOINT"   --outdir "${OUT_DIR:-outputs/plgmvs_dtu}" --interval_scale 1.06   --ndepths 8,8,4,4 --num_view 5 --conf 0.55 --attn_temp 2 "$@"
