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


parser = argparse.ArgumentParser(description='A PyTorch Implementation of MonoMVSNet')

parser.add_argument('--mode', default='train', help='train or test', choices=['train', 'test'])
parser.add_argument('--device', default='cuda', help='select model')
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument("--local-rank", dest="local_rank", type=int, default=0)
parser.add_argument('--seed', type=int, default=0, metavar='S', help='random seed')

# dataset
parser.add_argument('--dataset', default='dtu_yao_vit', help='select dataset')
parser.add_argument('--trainpath', default='your_dtu_training_path', help='train datapath')
parser.add_argument('--testpath', help='test datapath')
parser.add_argument('--trainlist', default='lists/dtu/train.txt', help='train list')
parser.add_argument('--testlist', default='lists/dtu/test.txt', help='test list')
parser.add_argument('--rt', default=True, help='robust training')
parser.add_argument('--max_h', type=int, default=512, help='testing max h')
parser.add_argument('--max_w', type=int, default=640, help='testing max w')
parser.add_argument('--interval_scale', type=float, default=1.06, help='the number of depth values')
parser.add_argument('--pin_m', action='store_true', help='data loader pin memory')

# training and testing
parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--lrepochs', type=str, default="10,12,14:2",
                    help='epoch ids to downscale lr and the downscale rate')
parser.add_argument('--wd', type=float, default=0.0, help='weight decay')
parser.add_argument('--batch_size', type=int, default=4, help='train batch size')
parser.add_argument('--loadckpt', default=None, help='load a specific checkpoint')
parser.add_argument('--logdir', default='./checkpoints/debug', help='the directory to save checkpoints/logs')
parser.add_argument('--resume', action='store_true', help='continue to train the model')
parser.add_argument('--summary_freq', type=int, default=100, help='print and summary frequency')
parser.add_argument('--save_freq', type=int, default=1, help='save checkpoint frequency')
parser.add_argument('--eval_freq', type=int, default=1, help='eval freq')
parser.add_argument('--dlossw', type=str, default="1,1,1,1", help='depth loss weight for different stage')
parser.add_argument('--use_raw_train', action='store_true', help='using 1200x1600 training')
parser.add_argument('--lr_scheduler', type=str, default='MS')

# network
parser.add_argument('--ndepths', type=str, default="8,8,4,4", help='ndepths')
parser.add_argument('--depth_inter_r', type=str, default="0.5,0.5,0.5,0.5", help='depth_intervals_ratio')
parser.add_argument("--fpn_base_channel", type=int, default=8)
parser.add_argument("--reg_channel", type=int, default=8)
parser.add_argument('--reg_mode', type=str, default="reg2d")
parser.add_argument('--group_cor', default=True, help='group correlation')
parser.add_argument('--group_cor_dim', type=str, default="8,8,4,4", help='group correlation dim')
parser.add_argument('--inverse_depth', default=True, help='inverse depth')

parser.add_argument('--mono_sampling', default=True, help='mono depth guide depth sampling')
parser.add_argument('--edge_guide', default=True, help='edge guide mono_sampling')
parser.add_argument('--attention', default=True, help='pyramid mono feature')
parser.add_argument('--n_views', default=5, help='training views')

parser.add_argument('--agg_type', type=str, default="ConvBnReLU3D", help='cost regularization type')
parser.add_argument('--arch_mode', type=str, default="fpn")
parser.add_argument('--attn_temp', type=float, default=2)


cudnn.benchmark = True
num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
is_distributed = num_gpus > 1


def unpack_model_loss(loss_result):
    """
    兼容两种 dtu_loss 返回格式：

    原始版本：
        loss, stage_ce_loss, stage_mono_loss, range_err_ratio

    加 refiner 版本：
        loss, stage_ce_loss, stage_mono_loss, stage_refine_loss, range_err_ratio
    """
    if len(loss_result) == 5:
        loss, stage_ce_loss, stage_mono_loss, stage_refine_loss, range_err_ratio = loss_result
    elif len(loss_result) == 4:
        loss, stage_ce_loss, stage_mono_loss, range_err_ratio = loss_result

        if len(stage_ce_loss) > 0:
            device = stage_ce_loss[0].device
            dtype = stage_ce_loss[0].dtype
        else:
            device = loss.device
            dtype = loss.dtype

        stage_refine_loss = [
            torch.zeros((), dtype=dtype, device=device)
            for _ in range(len(stage_ce_loss))
        ]
    else:
        raise ValueError("model_loss should return 4 or 5 values, but got {}".format(len(loss_result)))

    return loss, stage_ce_loss, stage_mono_loss, stage_refine_loss, range_err_ratio


def build_scalar_outputs(loss, stage_ce_loss, stage_mono_loss, stage_refine_loss,
                         range_err_ratio, depth_est, depth_gt, mask):
    """
    统一构建日志输出。
    会打印和保存：
    - total loss
    - 每个 stage 的 ce loss
    - 每个 stage 的 mono loss
    - 每个 stage 的 refine loss
    - 每个 stage 的 range error
    - depth metrics
    """
    scalar_outputs = {
        "loss": loss,
        "total_loss": loss,

        "s0_ce_loss": stage_ce_loss[0],
        "s1_ce_loss": stage_ce_loss[1],
        "s2_ce_loss": stage_ce_loss[2],
        "s3_ce_loss": stage_ce_loss[3],

        "s0_mono_loss": stage_mono_loss[0],
        "s1_mono_loss": stage_mono_loss[1],
        "s2_mono_loss": stage_mono_loss[2],
        "s3_mono_loss": stage_mono_loss[3],

        "s0_refine_loss": stage_refine_loss[0],
        "s1_refine_loss": stage_refine_loss[1],
        "s2_refine_loss": stage_refine_loss[2],
        "s3_refine_loss": stage_refine_loss[3],

        "refine_loss_stage1": stage_refine_loss[0],
        "refine_loss_stage2": stage_refine_loss[1],
        "refine_loss_stage3": stage_refine_loss[2],
        "refine_loss_stage4": stage_refine_loss[3],

        "s0_range_err_ratio": range_err_ratio[0],
        "s1_range_err_ratio": range_err_ratio[1],
        "s2_range_err_ratio": range_err_ratio[2],
        "s3_range_err_ratio": range_err_ratio[3],

        "range_err_stage1": range_err_ratio[0],
        "range_err_stage2": range_err_ratio[1],
        "range_err_stage3": range_err_ratio[2],
        "range_err_stage4": range_err_ratio[3],

        "abs_depth_error": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5),
        "thres2mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 2),
        "thres4mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 4),
        "thres8mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 8),
    }

    return scalar_outputs


def print_loss_log(prefix, epoch_idx, args, batch_idx, loader_len, lr, loss, scalar_outputs, start_time):
    print(
        "Epoch {}/{}, Iter {}/{}, lr {:.6f}, {} loss = {:.3f}, "
        "ce_loss = {:.3f}, {:.3f}, {:.3f}, {:.3f}, "
        "mono_loss = {:.3f}, {:.3f}, {:.3f}, {:.3f}, "
        "refine_loss = {:.6f}, {:.6f}, {:.6f}, {:.6f}, "
        "range_err = {:.3f}, {:.3f}, {:.3f}, {:.3f}, "
        "abs = {:.3f}, thres2 = {:.3f}, thres4 = {:.3f}, thres8 = {:.3f}, "
        "time = {:.3f}".format(
            epoch_idx, args.epochs, batch_idx, loader_len,
            lr,
            prefix,
            loss,

            scalar_outputs["s0_ce_loss"],
            scalar_outputs["s1_ce_loss"],
            scalar_outputs["s2_ce_loss"],
            scalar_outputs["s3_ce_loss"],

            scalar_outputs["s0_mono_loss"],
            scalar_outputs["s1_mono_loss"],
            scalar_outputs["s2_mono_loss"],
            scalar_outputs["s3_mono_loss"],

            scalar_outputs["s0_refine_loss"],
            scalar_outputs["s1_refine_loss"],
            scalar_outputs["s2_refine_loss"],
            scalar_outputs["s3_refine_loss"],

            scalar_outputs["s0_range_err_ratio"],
            scalar_outputs["s1_range_err_ratio"],
            scalar_outputs["s2_range_err_ratio"],
            scalar_outputs["s3_range_err_ratio"],

            scalar_outputs["abs_depth_error"],
            scalar_outputs["thres2mm_error"],
            scalar_outputs["thres4mm_error"],
            scalar_outputs["thres8mm_error"],

            time.time() - start_time
        )
    )


# main function
def train(model, model_loss, optimizer, TrainImgLoader, TestImgLoader, start_epoch, args):
    milestones = [len(TrainImgLoader) * int(epoch_idx) for epoch_idx in args.lrepochs.split(':')[0].split(',')]
    lr_gamma = 1 / float(args.lrepochs.split(':')[1])

    if args.lr_scheduler == 'MS':
        lr_scheduler = WarmupMultiStepLR(
            optimizer,
            milestones,
            gamma=lr_gamma,
            warmup_factor=1.0 / 3,
            warmup_iters=500,
            last_epoch=len(TrainImgLoader) * start_epoch - 1
        )
    elif args.lr_scheduler == 'cos':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(args.epochs * len(TrainImgLoader)),
            eta_min=0
        )
    elif args.lr_scheduler == 'onecycle':
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr,
            total_steps=int(args.epochs * len(TrainImgLoader))
        )
    else:
        raise NotImplementedError("Unknown lr_scheduler: {}".format(args.lr_scheduler))

    for epoch_idx in range(start_epoch, args.epochs):
        print('Epoch {}:'.format(epoch_idx))
        global_step = len(TrainImgLoader) * epoch_idx

        for batch_idx, sample in enumerate(TrainImgLoader):
            start_time = time.time()
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            do_summary = global_step % args.summary_freq == 0

            loss, scalar_outputs, image_outputs = train_sample(
                model, model_loss, optimizer, sample, args
            )

            lr_scheduler.step()

            if (not is_distributed) or (dist.get_rank() == 0):
                if do_summary:
                    save_scalars(logger, 'train', scalar_outputs, global_step)
                    save_images(logger, 'train', image_outputs, global_step)

                    print_loss_log(
                        prefix="train",
                        epoch_idx=epoch_idx,
                        args=args,
                        batch_idx=batch_idx,
                        loader_len=len(TrainImgLoader),
                        lr=optimizer.param_groups[0]["lr"],
                        loss=loss,
                        scalar_outputs=scalar_outputs,
                        start_time=start_time
                    )

                del scalar_outputs, image_outputs

        # checkpoint
        if (not is_distributed) or (dist.get_rank() == 0):
            if (epoch_idx + 1) % args.save_freq == 0:
                model_state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
                torch.save({
                    'epoch': epoch_idx,
                    'model': model_state,
                    'optimizer': optimizer.state_dict()
                }, "{}/finalmodel_{}.ckpt".format(args.logdir, epoch_idx))

        gc.collect()

        # testing
        if (epoch_idx % args.eval_freq == 0) or (epoch_idx == args.epochs - 1):
            avg_test_scalars = DictAverageMeter()

            for batch_idx, sample in enumerate(TestImgLoader):
                start_time = time.time()
                global_step = len(TestImgLoader) * epoch_idx + batch_idx
                do_summary = global_step % args.summary_freq == 0

                loss, scalar_outputs, image_outputs = test_sample_depth(
                    model, model_loss, sample, args
                )

                if (not is_distributed) or (dist.get_rank() == 0):
                    if do_summary:
                        save_scalars(logger, 'test', scalar_outputs, global_step)
                        save_images(logger, 'test', image_outputs, global_step)

                        print_loss_log(
                            prefix="test",
                            epoch_idx=epoch_idx,
                            args=args,
                            batch_idx=batch_idx,
                            loader_len=len(TestImgLoader),
                            lr=optimizer.param_groups[0]["lr"],
                            loss=loss,
                            scalar_outputs=scalar_outputs,
                            start_time=start_time
                        )

                    avg_test_scalars.update(scalar_outputs)
                    del scalar_outputs, image_outputs

            if (not is_distributed) or (dist.get_rank() == 0):
                save_scalars(logger, 'fulltest', avg_test_scalars.mean(), global_step)
                print("avg_test_scalars:", avg_test_scalars.mean())

            gc.collect()


def test(model, model_loss, TestImgLoader, args):
    avg_test_scalars = DictAverageMeter()

    for batch_idx, sample in enumerate(TestImgLoader):
        start_time = time.time()
        loss, scalar_outputs, image_outputs = test_sample_depth(
            model, model_loss, sample, args
        )

        avg_test_scalars.update(scalar_outputs)
        del scalar_outputs, image_outputs

        if (not is_distributed) or (dist.get_rank() == 0):
            print(
                'Iter {}/{}, test loss = {:.3f}, '
                'refine_loss = {:.6f}, {:.6f}, {:.6f}, {:.6f}, '
                'time = {:3f}'.format(
                    batch_idx,
                    len(TestImgLoader),
                    loss,
                    avg_test_scalars.mean()["s0_refine_loss"],
                    avg_test_scalars.mean()["s1_refine_loss"],
                    avg_test_scalars.mean()["s2_refine_loss"],
                    avg_test_scalars.mean()["s3_refine_loss"],
                    time.time() - start_time
                )
            )

            if batch_idx % 100 == 0:
                print("Iter {}/{}, test results = {}".format(
                    batch_idx,
                    len(TestImgLoader),
                    avg_test_scalars.mean()
                ))

    if (not is_distributed) or (dist.get_rank() == 0):
        print("final", avg_test_scalars.mean())


def train_sample(model, model_loss, optimizer, sample, args):
    model.train()
    optimizer.zero_grad()

    sample_cuda = tocuda(sample)
    depth_gt_ms = sample_cuda["depth"]
    mask_ms = sample_cuda["mask"]

    num_stage = len([int(nd) for nd in args.ndepths.split(",") if nd])
    depth_gt = depth_gt_ms["stage{}".format(num_stage)]
    mask = mask_ms["stage{}".format(num_stage)]

    outputs = model(
        sample_cuda["imgs"],
        sample_cuda["imgs_raw"],
        sample_cuda["proj_matrices"],
        sample_cuda["depth_values"]
    )

    depth_est = outputs["depth"]

    loss_result = model_loss(
        outputs,
        depth_gt_ms,
        mask_ms,
        stage_lw=[float(e) for e in args.dlossw.split(",") if e],
        inverse_depth=args.inverse_depth
    )

    loss, stage_ce_loss, stage_mono_loss, stage_refine_loss, range_err_ratio = unpack_model_loss(loss_result)

    loss.backward()
    optimizer.step()

    scalar_outputs = build_scalar_outputs(
        loss=loss,
        stage_ce_loss=stage_ce_loss,
        stage_mono_loss=stage_mono_loss,
        stage_refine_loss=stage_refine_loss,
        range_err_ratio=range_err_ratio,
        depth_est=depth_est,
        depth_gt=depth_gt,
        mask=mask
    )

    image_outputs = {
        "depth_est": depth_est * mask,
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

    outputs = model_eval(
        sample_cuda["imgs"],
        sample_cuda["imgs_raw"],
        sample_cuda["proj_matrices"],
        sample_cuda["depth_values"]
    )

    depth_est = outputs["depth"]

    loss_result = model_loss(
        outputs,
        depth_gt_ms,
        mask_ms,
        stage_lw=[float(e) for e in args.dlossw.split(",") if e],
        inverse_depth=args.inverse_depth
    )

    loss, stage_ce_loss, stage_mono_loss, stage_refine_loss, range_err_ratio = unpack_model_loss(loss_result)

    scalar_outputs = build_scalar_outputs(
        loss=loss,
        stage_ce_loss=stage_ce_loss,
        stage_mono_loss=stage_mono_loss,
        stage_refine_loss=stage_refine_loss,
        range_err_ratio=range_err_ratio,
        depth_est=depth_est,
        depth_gt=depth_gt,
        mask=mask
    )

    image_outputs = {
        "depth_est": depth_est * mask,
        "depth_est_nomask": depth_est,
        "depth_gt": sample["depth"]["stage1"],
        "ref_img": sample["imgs"][0],
        "mask": sample["mask"]["stage1"],
        "errormap": (depth_est - depth_gt).abs() * mask
    }

    if is_distributed:
        scalar_outputs = reduce_scalar_outputs(scalar_outputs)

    return tensor2float(scalar_outputs["loss"]), tensor2float(scalar_outputs), tensor2numpy(image_outputs)


if __name__ == '__main__':
    args = parser.parse_args()

    if args.resume:
        assert args.mode == "train"
        assert args.loadckpt is None

    if args.testpath is None:
        args.testpath = args.trainpath

    if is_distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl",
            init_method="env://"
        )
        synchronize()

    set_random_seed(args.seed)
    device = torch.device(args.device)

    if (not is_distributed) or (dist.get_rank() == 0):
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
    model = MonoMVSNet(
        arch_mode=args.arch_mode,
        reg_net=args.reg_mode,
        num_stage=4,
        fpn_base_channel=args.fpn_base_channel,
        reg_channel=args.reg_channel,
        stage_splits=[int(n) for n in args.ndepths.split(",")],
        depth_interals_ratio=[float(ir) for ir in args.depth_inter_r.split(",")],
        group_cor=args.group_cor,
        group_cor_dim=[int(n) for n in args.group_cor_dim.split(",")],
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
    model_loss = dtu_loss

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.wd
    )

    # load parameters
    start_epoch = 0

    if args.resume:
        saved_models = [fn for fn in os.listdir(args.logdir) if fn.endswith(".ckpt")]
        saved_models = sorted(saved_models, key=lambda x: int(x.split('_')[-1].split('.')[0]))

        loadckpt = os.path.join(args.logdir, saved_models[-1])
        print("resuming", loadckpt)

        state_dict = torch.load(loadckpt, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict['model'], strict=False)
        optimizer.load_state_dict(state_dict['optimizer'])
        start_epoch = state_dict['epoch'] + 1

    elif args.loadckpt:
        print("loading model {}".format(args.loadckpt))
        state_dict = torch.load(args.loadckpt, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict['model'], strict=False)

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
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )
    else:
        if torch.cuda.is_available():
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)

    # dataset, dataloader
    MVSDataset = find_dataset_def(args.dataset)

    if args.dataset.startswith('dtu'):
        train_dataset = MVSDataset(
            args.trainpath,
            args.trainlist,
            "train",
            args.n_views,
            args.interval_scale,
            rt=args.rt,
            use_raw_train=args.use_raw_train
        )

        test_dataset = MVSDataset(
            args.testpath,
            args.testlist,
            "val",
            args.n_views,
            args.interval_scale
        )

    elif args.dataset.startswith('blendedmvs'):
        train_dataset = MVSDataset(
            args.trainpath,
            args.trainlist,
            "train",
            7,
            robust_train=args.rt
        )

        test_dataset = MVSDataset(
            args.testpath,
            args.testlist,
            "val",
            7
        )

    else:
        raise NotImplementedError("Unknown dataset: {}".format(args.dataset))

    if is_distributed:
        train_sampler = torch.utils.data.DistributedSampler(
            train_dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank()
        )

        test_sampler = torch.utils.data.DistributedSampler(
            test_dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank()
        )

        TrainImgLoader = DataLoader(
            train_dataset,
            args.batch_size,
            sampler=train_sampler,
            num_workers=1,
            drop_last=True,
            pin_memory=args.pin_m
        )

        TestImgLoader = DataLoader(
            test_dataset,
            args.batch_size,
            sampler=test_sampler,
            num_workers=1,
            drop_last=False,
            pin_memory=args.pin_m
        )

    else:
        TrainImgLoader = DataLoader(
            train_dataset,
            args.batch_size,
            shuffle=True,
            num_workers=4,
            drop_last=True,
            pin_memory=args.pin_m
        )

        TestImgLoader = DataLoader(
            test_dataset,
            args.batch_size,
            shuffle=False,
            num_workers=1,
            drop_last=False,
            pin_memory=args.pin_m
        )

    if args.mode == "train":
        train(model, model_loss, optimizer, TrainImgLoader, TestImgLoader, start_epoch, args)

    elif args.mode == "test":
        test(model, model_loss, TestImgLoader, args)

    else:
        raise NotImplementedError
