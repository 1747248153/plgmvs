#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
EXPERIMENT="$ROOT/experiments/blended_tnt"
: "${TNT_ROOT:?请设置 TNT_ROOT 为 intermediate 场景父目录}"
: "${PLG_CHECKPOINT:?请设置 PLG_CHECKPOINT 为 TNT 版 PLG checkpoint 的绝对路径}"
if [[ ! -e "$EXPERIMENT/pre_trained_weights" ]]; then
  ln -s ../../pre_trained_weights "$EXPERIMENT/pre_trained_weights"
fi
cd "$EXPERIMENT"
exec "${PYTHON:-python}" test_dypcd_tnt_inter.py --dataset tanks --batch_size 1   --testpath "$TNT_ROOT" --testlist lists/tnt/inter_all.txt --loadckpt "$PLG_CHECKPOINT"   --outdir "${OUT_DIR:-outputs/plgmvs_tnt}" --interval_scale 1.06   --ndepths 8,8,4,4 --depth_inter_r 0.5,0.5,0.5,0.5 --num_view 21 --fusion_views 11   --num_worker 2 --group_cor --group_cor_dim 8,8,4,4 --inverse_depth --reg_mode reg2d   --reg_channel 8 --fpn_base_channel 8 --agg_type ConvBnReLU3D --arch_mode fpn --attn_temp 2 "$@"
