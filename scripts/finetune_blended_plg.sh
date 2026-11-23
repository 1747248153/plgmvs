#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
EXPERIMENT="$ROOT/experiments/blended_tnt"
: "${BLENDED_ROOT:?请设置 BLENDED_ROOT}"
: "${DTU_CHECKPOINT:?请设置 DTU_CHECKPOINT 为完整 PLG DTU 权重的绝对路径}"
if [[ ! -e "$EXPERIMENT/pre_trained_weights" ]]; then
  ln -s ../../pre_trained_weights "$EXPERIMENT/pre_trained_weights"
fi
cd "$EXPERIMENT"
exec "${PYTHON:-python}" train_bld.py --mode train --dataset blendedmvs   --trainpath "$BLENDED_ROOT" --trainlist lists/blendedmvs/train.txt --testlist lists/blendedmvs/val.txt   --loadckpt "$DTU_CHECKPOINT" --logdir "${LOG_DIR:-checkpoints/plg_blended_v4}"   --epochs 6 --lr 0.00005 --backbone_lr_scale 0.2 --refiner_warmup_epochs 1 --reset_refiner_head   --lrepochs 3,5:2 --lr_scheduler MS --wd 0.0001 --batch_size 1 --training_views 9   --summary_freq 100 --save_freq 1 --eval_freq 1 --max_h 576 --max_w 768 --interval_scale 1.06   --ndepths 8,8,4,4 --depth_inter_r 0.5,0.5,0.5,0.5 --group_cor True --group_cor_dim 8,8,4,4   --inverse_depth True --reg_mode reg2d --reg_channel 8 --fpn_base_channel 8 --agg_type ConvBnReLU3D   --arch_mode fpn --attn_temp 2 --mono_sampling True --edge_guide True --attention True   --refine_loss_weight 0.1 --grad_clip 1.0 --freeze_bn "$@"
