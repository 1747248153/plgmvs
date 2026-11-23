import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from models.module import *
from models.depth_anything_v2.dpt import DepthAnythingV2
from models.ted import TED
from models.FMT_with_CVPE import FMT_with_pathway
from models.depth_refine_sgformer import SGFormerDepthRefiner


class MonoMVSNet(nn.Module):
    def __init__(
        self,
        arch_mode="fpn",
        reg_net='reg2d',
        num_stage=4,
        fpn_base_channel=8,
        reg_channel=8,
        stage_splits=[8, 8, 4, 4],
        depth_interals_ratio=[0.5, 0.5, 0.5, 0.5],
        group_cor=False,
        group_cor_dim=[8, 8, 8, 8],
        inverse_depth=False,
        agg_type='ConvBnReLU3D',
        attn_temp=2,
        attn_fuse_d=True,
        mono_sampling=True,
        edge_guide=True,
        attention=True,
        max_h=512,
        max_w=640,
    ):
        super(MonoMVSNet, self).__init__()

        self.arch_mode = arch_mode
        self.num_stage = num_stage
        self.depth_interals_ratio = depth_interals_ratio
        self.group_cor = group_cor
        self.group_cor_dim = group_cor_dim
        self.inverse_depth = inverse_depth
        self.mono_sampling = mono_sampling

        if self.mono_sampling:
            self.edge_guide = edge_guide
            self.edge_thres = 0.8
        else:
            self.edge_guide = False
            self.edge_thres = 0.8

        self.attention = attention
        if self.attention:
            self.transformer = FMT_with_pathway()
            self.pe_num = 8

        if arch_mode == "fpn":
            self.feature = FPN4(base_channels=fpn_base_channel, gn=False)
        else:
            raise NotImplementedError("Only fpn arch_mode is supported.")

        self.stage_splits = stage_splits
        self.reg = nn.ModuleList()
        self.stagenet = stagenet(inverse_depth, attn_fuse_d, attn_temp)

        if reg_net == 'reg3d':
            self.down_size = [3, 3, 2, 2]

        for idx in range(num_stage):
            if self.group_cor:
                in_dim = group_cor_dim[idx]
            else:
                in_dim = self.feature.out_channels[idx]

            if reg_net == 'reg2d':
                self.reg.append(
                    reg2d(
                        input_channel=in_dim,
                        base_channel=reg_channel,
                        conv_name=agg_type
                    )
                )
            elif reg_net == 'reg3d':
                self.reg.append(
                    reg3d(
                        in_channels=1,
                        base_channels=reg_channel,
                        down_size=self.down_size[idx]
                    )
                )
            else:
                raise NotImplementedError("Unsupported reg_net: {}".format(reg_net))

        model_configs = {
            'vits': {
                'encoder': 'vits',
                'features': 64,
                'out_channels': [48, 96, 192, 384]
            },
            'vitb': {
                'encoder': 'vitb',
                'features': 128,
                'out_channels': [96, 192, 384, 768]
            },
            'vitl': {
                'encoder': 'vitl',
                'features': 256,
                'out_channels': [256, 512, 1024, 1024]
            },
            'vitg': {
                'encoder': 'vitg',
                'features': 384,
                'out_channels': [1536, 1536, 1536, 1536]
            }
        }

        encoder = 'vits'
        model = DepthAnythingV2(**model_configs[encoder])
        model.load_state_dict(
            torch.load(
                f'pre_trained_weights/depth_anything_v2_{encoder}.pth',
                map_location='cpu',
                weights_only=True
            )
        )

        self.mono = model
        self.mono.eval()
        self.max_h = max_h
        self.max_w = max_w

        for param in self.mono.pretrained.parameters():
            param.requires_grad = False

        for param in self.mono.depth_head.parameters():
            param.requires_grad = False

        self.vit_channel = model_configs[encoder]['out_channels'][-1]

        self.vit_down = nn.Sequential(
            nn.Conv2d(self.vit_channel, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU(),

            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),

            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU()
        )

        if self.edge_guide:
            edge_detection_model = TED()
            edge_detection_model.load_state_dict(
                torch.load('pre_trained_weights/TEED_model.pth'),
                strict=True
            )
            self.edge = edge_detection_model

        self.depth_refiner = SGFormerDepthRefiner(
            in_channels=4,
            embed_dim=8,
            num_heads=1,
            num_layers=1,
            residual_scale=0.05,
            refine_downsample=16
        )

    @staticmethod
    def _resize_3d_map(x, size):
        if x is None:
            return None

        if x.dim() == 3:
            if x.shape[-2:] != size:
                x = F.interpolate(
                    x.unsqueeze(1),
                    size=size,
                    mode="bilinear",
                    align_corners=False
                ).squeeze(1)
            return x

        if x.dim() == 4 and x.shape[1] == 1:
            if x.shape[-2:] != size:
                x = F.interpolate(
                    x,
                    size=size,
                    mode="bilinear",
                    align_corners=False
                )
            return x

        return x

    @staticmethod
    def _resize_4d_map(x, size):
        if x is None:
            return None

        if x.shape[-2:] != size:
            x = F.interpolate(
                x,
                size=size,
                mode="bilinear",
                align_corners=False
            )

        return x

    def _align_stage_outputs(self, outputs_stage, size):
        for key in [
            'depth',
            'depth_reg',
            'raw_depth',
            'raw_depth_reg',
            'depth_before_refiner',
            'refined_depth',
            'photometric_confidence',
            'inverse_min_depth',
            'inverse_max_depth',
        ]:
            if key in outputs_stage and torch.is_tensor(outputs_stage[key]):
                outputs_stage[key] = self._resize_3d_map(outputs_stage[key], size)

        return outputs_stage

    def _get_fpn_conv3_size(self, ref_img):
        feature_training = self.feature.training
        self.feature.eval()

        with torch.no_grad():
            ref_conv0 = self.feature.conv0(ref_img)
            ref_conv1 = self.feature.conv1(ref_conv0)
            ref_conv2 = self.feature.conv2(ref_conv1)
            ref_conv3 = self.feature.conv3(ref_conv2)

        if feature_training:
            self.feature.train()

        return ref_conv3.shape[-2], ref_conv3.shape[-1]


    def _run_edge_safe(self, raw_img, target_size):
        """
        T&T 尺寸安全版 TED edge。

        修复内容：
        1. imgs_raw 可能是 Byte / uint8，先转 float。
        2. 如果数值范围是 0~255，则归一化到 0~1。
        3. 先 resize 到 32 的倍数，避免 TED 内部 torch.cat 尺寸不一致。
        4. TED 输出后再 resize 回网络输入尺寸。
        """
        if raw_img is None:
            return None

        if raw_img.dim() == 3:
            raw_img = raw_img.unsqueeze(0)

        raw_img = raw_img.float()

        if raw_img.max() > 2.0:
            raw_img = raw_img / 255.0

        safe_h = int(target_size[0])
        safe_w = int(target_size[1])

        safe_h = max(32, (safe_h // 32) * 32)
        safe_w = max(32, (safe_w // 32) * 32)

        edge_input = F.interpolate(
            raw_img,
            size=(safe_h, safe_w),
            mode="bilinear",
            align_corners=False
        )

        with torch.no_grad():
            edge = self.edge(edge_input)
            edge = torch.sigmoid(edge)

        edge = F.interpolate(
            edge,
            size=target_size,
            mode="bilinear",
            align_corners=False
        )

        return edge


    def forward(self, imgs, imgs_raw, proj_matrices, depth_values):
        ref_img = imgs[0]
        _, _, ori_h, ori_w = ref_img.shape

        resize_h = ori_h // 14 * 14
        resize_w = ori_w // 14 * 14

        ref_img_resize = F.interpolate(
            ref_img,
            (resize_h, resize_w),
            mode="bilinear",
            align_corners=True
        )

        with torch.no_grad():
            mono_depth, mono_intermediate_features = self.mono.infer_mono(
                ref_img_resize,
                ori_h,
                ori_w
            )

            b, _, h, w = mono_depth.shape
            mono_depth = mono_depth.flatten(1)

            mono_depth_max = torch.max(mono_depth, dim=-1, keepdim=True)[0]
            mono_depth_min = torch.min(mono_depth, dim=-1, keepdim=True)[0]

            mono_depth = (
                mono_depth - mono_depth_min
            ) / (
                mono_depth_max - mono_depth_min + 1e-6
            )

            mono_depth = mono_depth.reshape(b, 1, h, w).contiguous()

        curr_features = (
            mono_intermediate_features[-1]
            .reshape(ref_img.shape[0], resize_h // 14, resize_w // 14, -1)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        target_h, target_w = self._get_fpn_conv3_size(ref_img)

        curr_features = F.interpolate(
            curr_features,
            (target_h, target_w),
            mode="bilinear",
            align_corners=True,
        )

        mono_feature = self.vit_down(curr_features)

        if self.edge_guide:
            ref_edge = self._run_edge_safe(
                imgs_raw[0],
                target_size=ref_img.shape[-2:]
            )
        else:
            ref_edge = None

        ref_outputs, src_outputs = self.feature(imgs, mono_feature)

        outputs = {}
        outputs_stage = None

        for stage_idx in range(self.num_stage):
            stage_name = "stage{}".format(stage_idx + 1)

            ref_features_stage = ref_outputs[stage_name]
            proj_matrices_stage = proj_matrices[stage_name]

            B, C, H, W = ref_features_stage.shape
            stage_size = (H, W)

            mono_depth_stage = self._resize_4d_map(mono_depth, stage_size)
            ref_edge_stage = self._resize_4d_map(ref_edge, stage_size) if ref_edge is not None else None

            if stage_idx == 0:
                if self.mono_sampling:
                    depth_hypo, mono_depth_scale = init_inverse_range_mono(
                        depth_values,
                        self.stage_splits[stage_idx],
                        imgs[0][0].device,
                        imgs[0][0].dtype,
                        H,
                        W,
                        mono_depth_stage,
                        ref_edge_stage,
                        self.edge_thres
                    )

                    if self.attention:
                        depth_hypo_cvpe = init_inverse_range(
                            depth_values,
                            self.pe_num,
                            imgs[0][0].device,
                            imgs[0][0].dtype,
                            H,
                            W
                        )

                        src_outputs = self.transformer(
                            ref_features_stage,
                            src_outputs,
                            proj_matrices_stage,
                            depth_hypo_cvpe
                        )
                else:
                    depth_hypo = init_inverse_range(
                        depth_values,
                        self.stage_splits[stage_idx],
                        imgs[0][0].device,
                        imgs[0][0].dtype,
                        H,
                        W
                    )

                    if self.attention:
                        depth_hypo_cvpe = init_inverse_range(
                            depth_values,
                            self.pe_num,
                            imgs[0][0].device,
                            imgs[0][0].dtype,
                            H,
                            W
                        )

                        src_outputs = self.transformer(
                            ref_features_stage,
                            src_outputs,
                            proj_matrices_stage,
                            depth_hypo_cvpe
                        )

            else:
                prev_inverse_min_depth = self._resize_3d_map(
                    outputs_stage['inverse_min_depth'].detach(),
                    stage_size
                )

                prev_inverse_max_depth = self._resize_3d_map(
                    outputs_stage['inverse_max_depth'].detach(),
                    stage_size
                )

                prev_depth = self._resize_3d_map(
                    outputs_stage['depth'].detach(),
                    stage_size
                )

                prev_confidence = self._resize_3d_map(
                    outputs_stage['photometric_confidence'].detach(),
                    stage_size
                )

                if self.mono_sampling:
                    depth_hypo, mono_depth_scale = schedule_inverse_range_mono(
                        prev_inverse_min_depth,
                        prev_inverse_max_depth,
                        self.stage_splits[stage_idx],
                        H,
                        W,
                        mono_depth_stage,
                        ref_edge_stage,
                        self.edge_thres,
                        prev_depth,
                        prev_confidence
                    )
                else:
                    depth_hypo = schedule_inverse_range(
                        prev_inverse_min_depth,
                        prev_inverse_max_depth,
                        self.stage_splits[stage_idx],
                        H,
                        W
                    )

            src_features_stage = [
                feat[stage_name]
                for feat in src_outputs
            ]

            fixed_src_features_stage = []
            for feat in src_features_stage:
                if feat.shape[-2:] != stage_size:
                    feat = F.interpolate(
                        feat,
                        size=stage_size,
                        mode="bilinear",
                        align_corners=True
                    )
                fixed_src_features_stage.append(feat)

            src_features_stage = fixed_src_features_stage

            outputs_stage = self.stagenet(
                ref_features_stage,
                src_features_stage,
                proj_matrices_stage,
                depth_hypo=depth_hypo,
                regnet=self.reg[stage_idx],
                group_cor=self.group_cor,
                group_cor_dim=self.group_cor_dim[stage_idx],
                split_itv=self.depth_interals_ratio[stage_idx]
            )

            outputs_stage = self._align_stage_outputs(outputs_stage, stage_size)

            outputs_stage['mono_depth'] = mono_depth_stage

            outputs_stage['raw_depth'] = outputs_stage['depth']
            outputs_stage['raw_depth_reg'] = outputs_stage['depth_reg']
            outputs_stage['depth_before_refiner'] = outputs_stage['depth_reg']

            refined_depth, depth_residual = self.depth_refiner(
                outputs_stage['depth_reg'],
                outputs_stage['photometric_confidence'],
                mono_depth_stage,
                ref_edge_stage
            )

            outputs_stage['refined_depth'] = refined_depth
            outputs_stage['depth_residual'] = depth_residual

            outputs_stage['depth'] = refined_depth
            outputs_stage['depth_reg'] = refined_depth

            outputs_stage = self._align_stage_outputs(outputs_stage, stage_size)

            outputs[stage_name] = outputs_stage
            outputs.update(outputs_stage)

        return outputs


def dtu_loss(inputs, depth_gt_ms, mask_ms, **kwargs):
    stage_lw = kwargs.get("stage_lw", [1, 1, 1, 1])
    threshold = [0.15, 0.15, 0.3, 0.3]
    inverse = kwargs.get("inverse_depth", False)
    refine_stage_lw = kwargs.get("refine_stage_lw", [0.01, 0.01, 0.02, 0.05])

    total_loss = torch.tensor(
        0.0,
        dtype=torch.float32,
        device=mask_ms["stage1"].device,
        requires_grad=False
    )

    stage_ce_loss = []
    stage_mono_loss = []
    stage_refine_loss = []
    range_err_ratio = []

    for stage_idx, (stage_inputs, stage_key) in enumerate(
        [(inputs[k], k) for k in inputs.keys() if "stage" in k]
    ):
        depth_reg = stage_inputs['depth_reg']
        hypo_depth = stage_inputs['hypo_depth']
        attn_weight = stage_inputs['attn_weight']
        confidence = stage_inputs['photometric_confidence']
        mono_depth = stage_inputs['mono_depth']

        mask = mask_ms[stage_key]
        mask = mask > 0.5

        conf_mask = confidence > threshold[stage_idx]
        mono_mask = torch.logical_or(mask, conf_mask)

        depth_gt = depth_gt_ms[stage_key]

        if inverse:
            depth_itv = (
                1 / hypo_depth[:, 2, :, :]
                - 1 / hypo_depth[:, 1, :, :]
            ).abs()

            mask_out_of_range = (
                (
                    1 / hypo_depth
                    - 1 / depth_gt.unsqueeze(1)
                ).abs()
                <= depth_itv.unsqueeze(1)
            ).sum(1) == 0
        else:
            depth_itv = (
                hypo_depth[:, 2, :, :]
                - hypo_depth[:, 1, :, :]
            ).abs()

            mask_out_of_range = (
                (
                    hypo_depth
                    - depth_gt.unsqueeze(1)
                ).abs()
                <= depth_itv.unsqueeze(1)
            ).sum(1) == 0

        range_err_ratio.append(mask_out_of_range[mask].float().mean())

        this_stage_ce_loss = cross_entropy_loss(
            mask,
            hypo_depth,
            depth_gt,
            attn_weight
        )

        if "refined_depth" in stage_inputs:
            refined_depth = stage_inputs["refined_depth"]
            this_stage_refine_loss = F.smooth_l1_loss(
                refined_depth[mask],
                depth_gt[mask],
                reduction="mean"
            )
        else:
            this_stage_refine_loss = torch.tensor(
                0.0,
                dtype=torch.float32,
                device=mask_ms["stage1"].device,
                requires_grad=False
            )

        this_refine_lw = refine_stage_lw[stage_idx]

        if stage_idx == 3:
            mono_depth = F.interpolate(
                mono_depth,
                size=depth_reg.shape[1:],
                mode='bilinear',
                align_corners=False
            )

            this_stage_mono_loss = relative_consistency_loss(
                depth_reg[mono_mask],
                mono_depth.squeeze(1)[mono_mask],
                sample_n=512 * (4 ** stage_idx)
            )

            total_loss = (
                total_loss
                + stage_lw[stage_idx] * this_stage_ce_loss
                + 0.01 * this_stage_mono_loss
                + this_refine_lw * this_stage_refine_loss
            )
        else:
            this_stage_mono_loss = torch.tensor(
                0.0,
                dtype=torch.float32,
                device=mask_ms["stage1"].device,
                requires_grad=False
            )

            total_loss = (
                total_loss
                + stage_lw[stage_idx] * this_stage_ce_loss
                + this_refine_lw * this_stage_refine_loss
            )

        stage_ce_loss.append(this_stage_ce_loss)
        stage_mono_loss.append(0.01 * this_stage_mono_loss)
        stage_refine_loss.append(this_stage_refine_loss)

    return total_loss, stage_ce_loss, stage_mono_loss, stage_refine_loss, range_err_ratio


def bld_loss(inputs, depth_gt_ms, mask_ms, **kwargs):
    stage_lw = kwargs.get("stage_lw", [1, 1, 1, 1])
    threshold = [0.15, 0.15, 0.3, 0.3]
    inverse = kwargs.get("inverse_depth", False)

    total_loss = torch.tensor(
        0.0,
        dtype=torch.float32,
        device=mask_ms["stage1"].device,
        requires_grad=False
    )

    stage_ce_loss = []
    range_err_ratio = []

    for stage_idx, (stage_inputs, stage_key) in enumerate(
        [(inputs[k], k) for k in inputs.keys() if "stage" in k]
    ):
        depth_pred = stage_inputs['depth']
        hypo_depth = stage_inputs['hypo_depth']
        attn_weight = stage_inputs['attn_weight']

        mono_depth = stage_inputs['mono_depth']
        confidence = stage_inputs['photometric_confidence']

        mask = mask_ms[stage_key]
        mask = mask > 0.5

        conf_mask = confidence > threshold[stage_idx]
        mono_mask = torch.logical_or(mask, conf_mask)

        depth_gt = depth_gt_ms[stage_key]

        if inverse:
            depth_itv = (
                1 / hypo_depth[:, 2, :, :]
                - 1 / hypo_depth[:, 1, :, :]
            ).abs()

            mask_out_of_range = (
                (
                    1 / hypo_depth
                    - 1 / depth_gt.unsqueeze(1)
                ).abs()
                <= depth_itv.unsqueeze(1)
            ).sum(1) == 0
        else:
            depth_itv = (
                hypo_depth[:, 2, :, :]
                - hypo_depth[:, 1, :, :]
            ).abs()

            mask_out_of_range = (
                (
                    hypo_depth
                    - depth_gt.unsqueeze(1)
                ).abs()
                <= depth_itv.unsqueeze(1)
            ).sum(1) == 0

        range_err_ratio.append(mask_out_of_range[mask].float().mean())

        this_stage_ce_loss = cross_entropy_loss(
            mask,
            hypo_depth,
            depth_gt,
            attn_weight
        )

        stage_ce_loss.append(this_stage_ce_loss)
        total_loss = total_loss + stage_lw[stage_idx] * this_stage_ce_loss

    depth_interval = hypo_depth[:, 0, :, :] - hypo_depth[:, 1, :, :]

    abs_err = torch.abs(depth_gt[mask] - depth_pred[mask])
    abs_err_scaled = abs_err / (depth_interval[mask] * 192. / 128.)

    epe = abs_err_scaled.mean()
    err3 = (abs_err_scaled < 3).float().mean()
    err1 = (abs_err_scaled < 1).float().mean()

    return total_loss, stage_ce_loss, range_err_ratio, epe, err3, err1


def cross_entropy_loss(mask_true, hypo_depth, depth_gt, attn_weight):
    B, D, H, W = attn_weight.shape

    valid_pixel_num = torch.sum(mask_true, dim=[1, 2]) + 1e-6

    gt_index_image = torch.argmin(
        torch.abs(hypo_depth - depth_gt.unsqueeze(1)),
        dim=1
    )

    gt_index_image = torch.mul(
        mask_true,
        gt_index_image.type(torch.float)
    )

    gt_index_image = torch.round(gt_index_image).type(torch.long).unsqueeze(1)

    gt_index_volume = torch.zeros(
        B,
        D,
        H,
        W
    ).type(mask_true.type()).scatter_(1, gt_index_image, 1)

    cross_entropy_image = -torch.sum(
        gt_index_volume * torch.log(attn_weight + 1e-6),
        dim=1
    ).squeeze(1)

    masked_cross_entropy_image = torch.mul(
        mask_true,
        cross_entropy_image
    )

    masked_cross_entropy = torch.sum(
        masked_cross_entropy_image,
        dim=[1, 2]
    )

    masked_cross_entropy = torch.mean(
        masked_cross_entropy / valid_pixel_num
    )

    return masked_cross_entropy


def relative_consistency_loss(mv_depth, mono_depth, sample_n=512):
    mv_depth = mv_depth.view(-1)
    mono_depth = mono_depth.view(-1)

    valid_num = mv_depth.shape[0]

    if valid_num < 2:
        return torch.tensor(
            0.0,
            dtype=mv_depth.dtype,
            device=mv_depth.device,
            requires_grad=True
        )

    sample_n = min(sample_n, valid_num // 2)

    selected_idxs = np.random.choice(
        valid_num,
        sample_n * 2,
        replace=False
    )

    sample_idx0 = selected_idxs[:sample_n]
    sample_idx1 = selected_idxs[sample_n:]

    mono_depth_0 = mono_depth[sample_idx0]
    mono_depth_1 = mono_depth[sample_idx1]

    mv_depth_0 = mv_depth[sample_idx0]
    mv_depth_1 = mv_depth[sample_idx1]

    mask = torch.where(mono_depth_0 < mono_depth_1, True, False)

    d0 = mv_depth_0 - mv_depth_1
    d1 = mv_depth_1 - mv_depth_0

    rc_loss = torch.zeros_like(mv_depth_0)
    rc_loss[mask] += d1[mask]
    rc_loss[~mask] += d0[~mask]

    return torch.mean(torch.clamp(rc_loss, min=0.0))
