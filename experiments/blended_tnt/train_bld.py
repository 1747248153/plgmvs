
# ===== compatibility patch for torch.distributed.launch / torchrun =====
# Convert "--local-rank=0" or "--local-rank 0" to "--local_rank=0"
# because this codebase originally defines argparse option as "--local_rank".
import sys as _sys_for_local_rank_patch
_new_argv = []
_skip_next = False
for i, _arg in enumerate(_sys_for_local_rank_patch.argv):
    if _skip_next:
        _skip_next = False
        continue
    if _arg.startswith("--local-rank="):
        _new_argv.append(_arg.replace("--local-rank=", "--local_rank=", 1))
    elif _arg == "--local-rank":
        _new_argv.append("--local_rank")
        if i + 1 < len(_sys_for_local_rank_patch.argv):
            _new_argv.append(_sys_for_local_rank_patch.argv[i + 1])
            _skip_next = True
    else:
        _new_argv.append(_arg)
_sys_for_local_rank_patch.argv = _new_argv
# ======================================================================

import argparse, os, sys, time, gc, datetime
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from datasets import find_dataset_def
from models import *
from utils import *
import torch.distributed as dist
from datasets.data_io import read_pfm, save_pfm

cudnn.benchmark = True

# os.environ["CUDA_VISIBLE_DEVICES"] = '2'

parser = argparse.ArgumentParser(description='A PyTorch Implementation of ET-MVSNet')

parser.add_argument('--mode', default='train', help='train or test', choices=['train', 'test'])
parser.add_argument('--device', default='cuda', help='select model')
parser.add_argument("--local_rank", type=int, default=0)
# parser.add_argument('--local-rank', dest='local_rank', type=int, help='node rank for distributed training')
parser.add_argument('--seed', type=int, default=0, metavar='S', help='random seed')

# dataset
parser.add_argument('--dataset', default='blendedmvs', help='select dataset')
parser.add_argument('--trainpath', default='your_blendedmvs_path', help='train datapath')
parser.add_argument('--testpath', help='test datapath')
parser.add_argument('--trainlist', default='lists/blendedmvs/train.txt', help='train list')
parser.add_argument('--testlist', default='lists/blendedmvs/val.txt',help='test list')
parser.add_argument('--rt', default=True,help='robust training')
parser.add_argument('--max_h', type=int, default=864, help='testing max h')
parser.add_argument('--max_w', type=int, default=1152, help='testing max w')
parser.add_argument('--interval_scale', type=float, default=1.06, help='the number of depth values')
parser.add_argument('--pin_m', action='store_true', help='data loader pin memory')

# training and testing
parser.add_argument('--epochs', type=int, default=16, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--lrepochs', type=str, default="6,8,10,12:2", help='epoch ids to downscale lr and the downscale rate')
parser.add_argument('--wd', type=float, default=0.0, help='weight decay')
parser.add_argument('--batch_size', type=int, default=2, help='train batch size')
parser.add_argument('--training_views', type=int, default=11, help='train batch size')
parser.add_argument('--loadckpt', default='checkpoints/dtu_train/dtu_best.ckpt', help='load a specific checkpoint')
parser.add_argument('--logdir', default='checkpoints/bld_ft', help='the directory to save checkpoints/logs')
parser.add_argument('--resume', action='store_true', help='continue to train the model')
parser.add_argument('--summary_freq', type=int, default=1, help='print and summary frequency')
parser.add_argument('--save_freq', type=int, default=1, help='save checkpoint frequency')
parser.add_argument('--eval_freq', type=int, default=1, help='eval freq')
parser.add_argument('--dlossw', type=str, default="1,1,1,1", help='depth loss weight for different stage')
parser.add_argument('--refine_loss_weight', type=float, default=0.1,
                    help='weight of interval-normalized refined-depth loss')
parser.add_argument('--grad_clip', type=float, default=1.0,
                    help='maximum gradient norm; non-positive disables clipping')
parser.add_argument('--freeze_bn', action='store_true',
                    help='freeze BatchNorm running statistics during small-batch fine-tuning')
parser.add_argument('--backbone_lr_scale', type=float, default=1.0,
                    help='backbone learning-rate multiplier relative to --lr')
parser.add_argument('--refiner_warmup_epochs', type=int, default=0,
                    help='initial epochs that update only depth_refiner parameters')
parser.add_argument('--reset_refiner_head', action='store_true',
                    help='zero the residual output head after initialization checkpoint loading')
parser.add_argument('--max_train_batches', type=int, default=0,
                    help='limit batches per epoch for a diagnostic pilot; 0 disables the limit')
parser.add_argument('--required_max_abs_depth_error', type=float, default=None,
                    help='only save bestmodel.ckpt when validation abs depth error is strictly below this baseline')
parser.add_argument('--use_raw_train', action='store_true',help='using 1200x1600 training')
parser.add_argument('--lr_scheduler', type=str, default='MS')

# network
parser.add_argument('--ndepths', type=str, default="8,8,4,4", help='ndepths')
parser.add_argument('--depth_inter_r', type=str, default="0.5,0.5,0.5,0.5", help='depth_intervals_ratio')
parser.add_argument("--fpn_base_channel", type=int, default=8)
parser.add_argument("--reg_channel", type=int, default=8)
parser.add_argument('--reg_mode', type=str, default="reg2d")
parser.add_argument('--group_cor', default=True, help='group correlation')
parser.add_argument('--group_cor_dim', type=str, default="8,8,4,4", help='group correlation dim')
parser.add_argument('--inverse_depth', default=True,help='inverse depth')
parser.add_argument('--agg_type', type=str, default="ConvBnReLU3D", help='cost regularization type')
parser.add_argument('--arch_mode', type=str, default="fpn")
parser.add_argument('--attn_temp', type=float, default=2)

parser.add_argument('--mono_sampling', default=True, help='mono depth guide depth sampling')
parser.add_argument('--edge_guide', default=True, help='edge guide mono_sampling')
parser.add_argument('--attention', default=True, help='pyramid mono feature')



num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
is_distributed = num_gpus > 1

# main function
def train(model, model_loss, optimizer, TrainImgLoader, TestImgLoader, start_epoch, args):
    # ===== scheduler resume patch =====
    # 当从 finalmodel_N.ckpt 续训但没有完整 optimizer state 时，
    # PyTorch scheduler 需要 param_group 里有 initial_lr。
    for _pg in optimizer.param_groups:
        if "initial_lr" not in _pg:
            _pg["initial_lr"] = _pg["lr"]
    # ==================================

    milestones = [len(TrainImgLoader) * int(epoch_idx) for epoch_idx in args.lrepochs.split(':')[0].split(',')]
    lr_gamma = 1 / float(args.lrepochs.split(':')[1])
    if args.lr_scheduler == 'MS':
        lr_scheduler = WarmupMultiStepLR(optimizer, milestones, gamma=lr_gamma, warmup_factor=1.0/3, warmup_iters=500,
                                                            last_epoch=len(TrainImgLoader) * start_epoch - 1)
    elif args.lr_scheduler == 'cos':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(args.epochs*len(TrainImgLoader)), eta_min=0)
    elif args.lr_scheduler == 'onecycle':
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr,total_steps=int(args.epochs*len(TrainImgLoader)))

    best_ckpt_path = os.path.join(args.logdir, "bestmodel.ckpt")
    best_abs_depth_error = (float(args.required_max_abs_depth_error)
                            if args.required_max_abs_depth_error is not None
                            else float("inf"))
    if args.required_max_abs_depth_error is not None:
        print("最佳模型硬门槛: abs_depth_error 必须严格小于 {:.9f}".format(
            args.required_max_abs_depth_error))
    if os.path.isfile(best_ckpt_path):
        best_state = torch.load(best_ckpt_path, map_location="cpu")
        best_abs_depth_error = float(best_state.get("best_abs_depth_error", float("inf")))
        print("已有最佳验证 abs_depth_error: {:.6f}".format(best_abs_depth_error))

    train_start_time = time.time()
    base_model = model.module
    original_requires_grad = {
        name: parameter.requires_grad
        for name, parameter in base_model.named_parameters()
    }
    for epoch_idx in range(start_epoch, args.epochs):
        epoch_start_time = time.time()
        refiner_only = epoch_idx < args.refiner_warmup_epochs
        for name, parameter in base_model.named_parameters():
            parameter.requires_grad = (
                original_requires_grad[name]
                and (not refiner_only or name.startswith("depth_refiner."))
            )
        print("训练范围: {}".format("仅 depth_refiner" if refiner_only else "主干 + depth_refiner"))
        print('Epoch {}:'.format(epoch_idx))
        global_step = len(TrainImgLoader) * epoch_idx

        # training
        for batch_idx, sample in enumerate(TrainImgLoader):
            if args.max_train_batches > 0 and batch_idx >= args.max_train_batches:
                print("Pilot batch 限制已达到: {}".format(args.max_train_batches))
                break
            start_time = time.time()
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            do_summary = global_step % args.summary_freq == 0
            loss, scalar_outputs, image_outputs = train_sample(model, model_loss, optimizer, sample, args)
            lr_scheduler.step()
            if (not is_distributed) or (dist.get_rank() == 0):
                if do_summary:
                    save_scalars(logger, 'train', scalar_outputs, global_step)
                    save_images(logger, 'train', image_outputs, global_step)
                    print(
                       "Epoch {}/{}, Iter {}/{}, lr {:.6f}, train loss = {:.3f}, ce_loss = {:.3f}, {:.3f}, {:.3f}, {:.3f}, refine_loss = {:.3f}, {:.3f}, {:.3f}, {:.3f}, range_err = {:.3f}, {:.3f}, {:.3f}, {:.3f}, epe:{:.6f}, err3:{:.6f}, err1:{:.6f}, time = {:.3f}, gpu_current_mb = {:.1f}, gpu_peak_mb = {:.1f}".format(
                           epoch_idx, args.epochs, batch_idx, len(TrainImgLoader),
                           optimizer.param_groups[0]["lr"],
                           loss,
                           scalar_outputs["s0_ce_loss"],
                           scalar_outputs["s1_ce_loss"],
                           scalar_outputs["s2_ce_loss"],
                           scalar_outputs["s3_ce_loss"],
                           scalar_outputs["s0_refine_loss"],
                           scalar_outputs["s1_refine_loss"],
                           scalar_outputs["s2_refine_loss"],
                           scalar_outputs["s3_refine_loss"],
                           scalar_outputs["s0_range_err_ratio"],
                           scalar_outputs["s1_range_err_ratio"],
                           scalar_outputs["s2_range_err_ratio"],
                           scalar_outputs["s3_range_err_ratio"],
                           scalar_outputs['epe'],
                           scalar_outputs['err3'],
                           scalar_outputs['err1'],
                           time.time() - start_time,
                           torch.cuda.memory_allocated() / 1024 ** 2,
                           torch.cuda.max_memory_allocated() / 1024 ** 2))
                del scalar_outputs, image_outputs

        # checkpoint
        if (not is_distributed) or (dist.get_rank() == 0):
            if (epoch_idx + 1) % args.save_freq == 0:
                # if epoch_idx == args.epochs - 1:
                torch.save({
                    'epoch': epoch_idx,
                    'model': model.module.state_dict(),
                    'optimizer': optimizer.state_dict()},
                    "{}/finalmodel_{}.ckpt".format(args.logdir,epoch_idx))

        gc.collect()

        # testing
        if (epoch_idx % args.eval_freq == 0) or (epoch_idx == args.epochs - 1):
            avg_test_scalars = DictAverageMeter()
            for batch_idx, sample in enumerate(TestImgLoader):
                start_time = time.time()
                global_step = len(TestImgLoader) * epoch_idx + batch_idx
                do_summary = global_step % args.summary_freq == 0
                loss, scalar_outputs, image_outputs = test_sample_depth(model, model_loss, sample, args)
                if (not is_distributed) or (dist.get_rank() == 0):
                    if do_summary:
                        save_scalars(logger, 'test', scalar_outputs, global_step)
                        save_images(logger, 'test', image_outputs, global_step)
                        print(
                            "Epoch {}/{}, Iter {}/{}, lr {:.6f}, test loss = {:.3f}, c_loss = {:.3f}, {:.3f}, {:.3f}, {:.3f}, range_err = {:.3f}, {:.3f}, {:.3f}, {:.3f}, epe:{:.6f}, err3:{:.6f}, err1:{:.6f}, time = {:.3f}".format(
                            epoch_idx, args.epochs, batch_idx, len(TestImgLoader),
                               optimizer.param_groups[0]["lr"], 
                           loss,
                           scalar_outputs["s0_ce_loss"],
                           scalar_outputs["s1_ce_loss"],
                           scalar_outputs["s2_ce_loss"],
                           scalar_outputs["s3_ce_loss"],
                           scalar_outputs["s0_range_err_ratio"],
                           scalar_outputs["s1_range_err_ratio"],
                           scalar_outputs["s2_range_err_ratio"],
                           scalar_outputs["s3_range_err_ratio"],
                           scalar_outputs['epe'],
                           scalar_outputs['err3'],
                           scalar_outputs['err1'],
                           time.time() - start_time))
                    avg_test_scalars.update(scalar_outputs)
                    del scalar_outputs, image_outputs

            if (not is_distributed) or (dist.get_rank() == 0):
                validation_scalars = avg_test_scalars.mean()
                save_scalars(logger, 'fulltest', validation_scalars, global_step)
                print("avg_test_scalars:", validation_scalars)
                current_abs_depth_error = float(validation_scalars["abs_depth_error"])
                if current_abs_depth_error < best_abs_depth_error:
                    best_abs_depth_error = current_abs_depth_error
                    torch.save({
                        'epoch': epoch_idx,
                        'model': model.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'best_abs_depth_error': best_abs_depth_error,
                    }, best_ckpt_path)
                    print("保存最佳模型: {} (epoch={}, abs_depth_error={:.6f})".format(
                        best_ckpt_path, epoch_idx, best_abs_depth_error))
            gc.collect()
        if (not is_distributed) or (dist.get_rank() == 0):
            print("Epoch {} time = {:.1f}s, total training time = {:.1f}s, gpu_peak_mb = {:.1f}".format(
                epoch_idx, time.time() - epoch_start_time, time.time() - train_start_time,
                torch.cuda.max_memory_allocated() / 1024 ** 2))


def test(model, model_loss, TestImgLoader, args):
    avg_test_scalars = DictAverageMeter()
    for batch_idx, sample in enumerate(TestImgLoader):
        start_time = time.time()
        loss, scalar_outputs, image_outputs = test_sample_depth(model, model_loss, sample, args)
        avg_test_scalars.update(scalar_outputs)
        del scalar_outputs, image_outputs
        if (not is_distributed) or (dist.get_rank() == 0):
            print('Iter {}/{}, test loss = {:.3f}, time = {:3f}'.format(batch_idx, len(TestImgLoader), loss,
                                                                        time.time() - start_time))
            if batch_idx % 100 == 0:
                print("Iter {}/{}, test results = {}".format(batch_idx, len(TestImgLoader), avg_test_scalars.mean()))
    if (not is_distributed) or (dist.get_rank() == 0):
        print("final", avg_test_scalars.mean())


def train_sample(model, model_loss, optimizer, sample, args):
    model.train()
    if args.freeze_bn:
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()
    optimizer.zero_grad()

    sample_cuda = tocuda(sample)
    depth_gt_ms = sample_cuda["depth"]
    mask_ms = sample_cuda["mask"]

    num_stage = len([int(nd) for nd in args.ndepths.split(",") if nd])
    depth_gt = depth_gt_ms["stage{}".format(num_stage)]
    mask = mask_ms["stage{}".format(num_stage)]

    outputs = model(sample_cuda["imgs"], sample_cuda["imgs_raw"], sample_cuda["proj_matrices"], sample_cuda["depth_values"])
    depth_est = outputs["depth"]

    loss, stage_ce_loss, stage_refine_loss, range_err_ratio, epe, err3, err1 = model_loss(
                                        outputs, depth_gt_ms, mask_ms, stage_lw=[float(e) for e in args.dlossw.split(",") if e], 
                                        inverse_depth=args.inverse_depth,
                                        refine_loss_weight=args.refine_loss_weight
                                        )
    loss.backward()
    if not getattr(args, "_gradient_audit_done", False):
        refiner_grad_params = []
        refiner_missing_grad_params = []
        refiner_grad_is_finite = True
        for name, parameter in model.module.named_parameters():
            if not name.startswith("depth_refiner.") or not parameter.requires_grad:
                continue
            if parameter.grad is None:
                refiner_missing_grad_params.append(name)
            else:
                refiner_grad_params.append(name)
                refiner_grad_is_finite = (refiner_grad_is_finite
                                          and bool(torch.isfinite(parameter.grad).all()))
        print("RefineAll 梯度审计: 有梯度 {}/{}, 全部有限={}, 缺失={}".format(
            len(refiner_grad_params),
            len(refiner_grad_params) + len(refiner_missing_grad_params),
            refiner_grad_is_finite,
            refiner_missing_grad_params))
        if not refiner_grad_params or not refiner_grad_is_finite:
            raise RuntimeError("RefineAll 梯度审计失败")
        args._gradient_audit_done = True
    if args.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()

    scalar_outputs = {"loss": loss,
                      "s0_ce_loss": stage_ce_loss[0],
                      "s1_ce_loss": stage_ce_loss[1],
                      "s2_ce_loss": stage_ce_loss[2],
                      "s3_ce_loss": stage_ce_loss[3],
                      "s0_refine_loss": stage_refine_loss[0],
                      "s1_refine_loss": stage_refine_loss[1],
                      "s2_refine_loss": stage_refine_loss[2],
                      "s3_refine_loss": stage_refine_loss[3],
                      "s0_range_err_ratio":range_err_ratio[0],
                      "s1_range_err_ratio":range_err_ratio[1],
                      "s2_range_err_ratio":range_err_ratio[2],
                      "s3_range_err_ratio":range_err_ratio[3],
                      "abs_depth_error": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5),
                      "thres2mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 2),
                      "thres4mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 4),
                      "thres8mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 8),
                      'epe':epe,
                      'err3':err3,
                      'err1':err1}

    image_outputs = {"depth_est": depth_est * mask,
                     "depth_est_nomask": depth_est,
                     "depth_gt": sample["depth"]["stage1"],
                     "ref_img": sample["imgs"][0],
                     "mask": sample["mask"]["stage1"],
                     "errormap": (depth_est - depth_gt).abs() * mask,
                     }

    if is_distributed:
        scalar_outputs = reduce_scalar_outputs(scalar_outputs)

    return tensor2float(scalar_outputs["loss"]), tensor2float(scalar_outputs), tensor2numpy(image_outputs)


@make_nograd_func
def test_sample_depth(model, model_loss, sample, args):
    if is_distributed:
        model_eval = model.module
    else:
        model_eval = model
    model_eval.eval()

    sample_cuda = tocuda(sample)
    depth_gt_ms = sample_cuda["depth"]
    mask_ms = sample_cuda["mask"]

    num_stage = len([int(nd) for nd in args.ndepths.split(",") if nd])
    depth_gt = depth_gt_ms["stage{}".format(num_stage)]
    mask = mask_ms["stage{}".format(num_stage)]

    outputs = model_eval(sample_cuda["imgs"], sample_cuda["imgs_raw"], sample_cuda["proj_matrices"], sample_cuda["depth_values"])
    depth_est = outputs["depth"]
    loss, stage_ce_loss, stage_refine_loss, range_err_ratio, epe, err3, err1 = model_loss(
                                        outputs, depth_gt_ms, mask_ms, stage_lw=[float(e) for e in args.dlossw.split(",") if e], 
                                        inverse_depth=args.inverse_depth,
                                        refine_loss_weight=args.refine_loss_weight
                                        )
    scalar_outputs = {"loss": loss,
                      "s0_ce_loss": stage_ce_loss[0],
                      "s1_ce_loss": stage_ce_loss[1],
                      "s2_ce_loss": stage_ce_loss[2],
                      "s3_ce_loss": stage_ce_loss[3],
                      "s0_refine_loss": stage_refine_loss[0],
                      "s1_refine_loss": stage_refine_loss[1],
                      "s2_refine_loss": stage_refine_loss[2],
                      "s3_refine_loss": stage_refine_loss[3],
                      "s0_range_err_ratio":range_err_ratio[0],
                      "s1_range_err_ratio":range_err_ratio[1],
                      "s2_range_err_ratio":range_err_ratio[2],
                      "s3_range_err_ratio":range_err_ratio[3],
                      "abs_depth_error": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5),
                      "thres2mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 2),
                      "thres4mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 4),
                      "thres8mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 8),
                      'epe':epe,
                      'err3':err3,
                      'err1':err1}

    image_outputs = {"depth_est": depth_est * mask,
                     "depth_est_nomask": depth_est,
                     "depth_gt": sample["depth"]["stage1"],
                     "ref_img": sample["imgs"][0],
                     "mask": sample["mask"]["stage1"],
                     "errormap": (depth_est - depth_gt).abs() * mask}

    if is_distributed:
        scalar_outputs = reduce_scalar_outputs(scalar_outputs)

    return tensor2float(scalar_outputs["loss"]), tensor2float(scalar_outputs), tensor2numpy(image_outputs)


if __name__ == '__main__':
    # parse arguments and check
    args = parser.parse_args()

    if args.resume:
        assert args.mode == "train"
        assert not args.loadckpt

    if args.testpath is None:
        args.testpath = args.trainpath

    if is_distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    set_random_seed(args.seed)
    device = torch.device(args.device)

    if (not is_distributed) or (dist.get_rank() == 0):
        # create logger for mode "train" and "testall"
        if args.mode == "train":
            if not os.path.isdir(args.logdir):
                os.makedirs(args.logdir)
            current_time_str = str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
            print("current time", current_time_str)
            print("creating new summary file")
            logger = SummaryWriter(args.logdir)
        print("argv:", sys.argv[1:])
        print_args(args)

    # model, optimizer
    model = MonoMVSNet(arch_mode=args.arch_mode, reg_net=args.reg_mode, num_stage=4,
                    fpn_base_channel=args.fpn_base_channel, reg_channel=args.reg_channel, 
                    stage_splits=[int(n) for n in args.ndepths.split(",")], 
                    depth_interals_ratio=[float(ir) for ir in args.depth_inter_r.split(",")],
                    group_cor=args.group_cor, group_cor_dim=[int(n) for n in args.group_cor_dim.split(",")],
                    inverse_depth=args.inverse_depth,
                    agg_type=args.agg_type,
                    attn_temp=args.attn_temp,
                    mono_sampling=args.mono_sampling,
                    edge_guide=args.edge_guide,
                    attention=args.attention,
                    max_h=args.max_h,
                    max_w=args.max_w,
                )

    model.to(device)
    model_loss = bld_loss

    refiner_params = []
    backbone_params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("depth_refiner."):
            refiner_params.append(parameter)
        else:
            backbone_params.append(parameter)
    optimizer = optim.Adam([
        {'params': backbone_params, 'lr': args.lr * args.backbone_lr_scale,
         'name': 'backbone'},
        {'params': refiner_params, 'lr': args.lr, 'name': 'depth_refiner'},
    ], betas=(0.9, 0.999), weight_decay=args.wd)
    print("优化器分组: backbone lr={}, depth_refiner lr={}".format(
        args.lr * args.backbone_lr_scale, args.lr))

    # load parameters
    start_epoch = 0
    if args.resume:
        import re
        saved_models = [fn for fn in os.listdir(args.logdir)
                        if re.fullmatch(r"finalmodel_\d+\.ckpt", fn)]
        if not saved_models:
            raise FileNotFoundError("--resume 指定的目录中没有 finalmodel_N.ckpt: {}".format(args.logdir))
        saved_models = sorted(saved_models, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        # use the latest checkpoint file
        loadckpt = os.path.join(args.logdir, saved_models[-1])
        print("resuming", loadckpt)
        state_dict = torch.load(loadckpt, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict['model'])
        optimizer.load_state_dict(state_dict['optimizer'])
        start_epoch = state_dict['epoch'] + 1
    elif args.loadckpt:
        # load checkpoint file specified by args.loadckpt
        print("loading model {}".format(args.loadckpt))
        state_dict = torch.load(args.loadckpt, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict['model'],strict=True)
        if args.reset_refiner_head:
            output_layer = model.depth_refiner.out_conv[-1]
            torch.nn.init.zeros_(output_layer.weight)
            if output_layer.bias is not None:
                torch.nn.init.zeros_(output_layer.bias)
            print("已清零 depth_refiner 残差输出头；初始 refined depth 等于 raw depth。")

    # ===== resume epoch from finalmodel_N.ckpt =====
    # 如果 loadckpt 是 finalmodel_5.ckpt，则从 epoch 6 继续训练
    import os as _os_resume_epoch_patch
    import re as _re_resume_epoch_patch
    _ckpt_name = _os_resume_epoch_patch.path.basename(args.loadckpt or "")
    _m_resume = _re_resume_epoch_patch.search(r"finalmodel_(\d+)\.ckpt", _ckpt_name)
    # A DTU finalmodel_N checkpoint passed via --loadckpt is initialization,
    # not a request to continue the BlendedMVS epoch counter.
    if args.resume and _m_resume is not None:
        start_epoch = int(_m_resume.group(1)) + 1
        print("[Resume Patch] loadckpt =", args.loadckpt)
        print("[Resume Patch] start_epoch set to", start_epoch)
    # ==============================================


    if (not is_distributed) or (dist.get_rank() == 0):
        print("start at epoch {}".format(start_epoch))
        total_params = sum([p.data.nelement() for p in model.parameters()]) / 1e6
        trainable_params = sum([p.data.nelement() for p in model.parameters() if p.requires_grad]) / 1e6
        print('Number of model parameters: {:.2f}M'.format(total_params))
        print('Number of model trainable parameters: {:.2f}M'.format(trainable_params))


    if is_distributed:
        if dist.get_rank() == 0:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank,
            find_unused_parameters=True,
        )
    else:
        if torch.cuda.is_available():
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)

    # dataset, dataloader
    MVSDataset = find_dataset_def(args.dataset)
    if args.dataset.startswith('dtu'):
        train_dataset = MVSDataset(args.trainpath, args.trainlist, "train", 5, args.interval_scale, rt=args.rt, use_raw_train=args.use_raw_train)
        test_dataset = MVSDataset(args.testpath, args.testlist, "val", 5, args.interval_scale)
    elif args.dataset.startswith('blendedmvs'):
        train_dataset = MVSDataset(args.trainpath, args.trainlist, "train", args.training_views, robust_train=args.rt)
        test_dataset = MVSDataset(args.testpath, args.testlist, "val", args.training_views)
    if is_distributed:
        train_sampler = torch.utils.data.DistributedSampler(train_dataset, num_replicas=dist.get_world_size(),
                                                            rank=dist.get_rank())
        test_sampler = torch.utils.data.DistributedSampler(test_dataset, num_replicas=dist.get_world_size(),
                                                           rank=dist.get_rank())

        TrainImgLoader = DataLoader(train_dataset, args.batch_size, sampler=train_sampler, num_workers=1,
                                    drop_last=True,
                                    pin_memory=args.pin_m)
        TestImgLoader = DataLoader(test_dataset, args.batch_size, sampler=test_sampler, num_workers=1, drop_last=False,
                                   pin_memory=args.pin_m)
    else:
        TrainImgLoader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=1, drop_last=True,
                                    pin_memory=args.pin_m)
        TestImgLoader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=1, drop_last=False,
                                   pin_memory=args.pin_m)


    if args.mode == "train":
        train(model, model_loss, optimizer, TrainImgLoader, TestImgLoader, start_epoch, args)
    elif args.mode == "test":
        test(model, model_loss, TestImgLoader, args)
    else:
        raise NotImplementedError
