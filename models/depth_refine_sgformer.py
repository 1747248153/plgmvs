import torch
import torch.nn as nn
import torch.nn.functional as F


class SGFormerDepthRefiner(nn.Module):
    """
    Full PLG-Refiner / Scheme 2.

    This version is compatible with checkpoints that contain:
        depth_refiner.global_attn.*
        depth_refiner.fuse.*

    Input:
        depth_reg + photometric_confidence + mono_depth + edge

    Structure:
        input_proj
        + local_branch
        + global_attn
        + fuse
        + out_conv

    Output:
        refined_depth = depth_reg + 0.05 * depth_residual
    """

    def __init__(
        self,
        in_channels=4,
        embed_dim=8,
        num_heads=1,
        num_layers=1,
        residual_scale=0.05,
        refine_downsample=16
    ):
        super(SGFormerDepthRefiner, self).__init__()

        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.residual_scale = residual_scale
        self.refine_downsample = refine_downsample

        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),

            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )

        self.local_branch = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),

            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )

        self.global_attn = SimpleGlobalAttention(embed_dim=embed_dim)

        self.fuse = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )

        self.out_conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, 1, kernel_size=3, padding=1),
        )

        print("[Restore] SGFormerDepthRefiner = Full old-compatible version: local_branch + global_attn + fuse.")

    @staticmethod
    def _to_4d(x):
        if x is None:
            return None
        if x.dim() == 3:
            x = x.unsqueeze(1)
        return x

    def forward(self, depth_reg, confidence, mono_depth, edge=None):
        depth_reg_4d = self._to_4d(depth_reg)
        conf_4d = self._to_4d(confidence)
        mono_4d = self._to_4d(mono_depth)

        B, _, H, W = depth_reg_4d.shape

        if conf_4d is None:
            conf_4d = torch.zeros_like(depth_reg_4d)
        elif conf_4d.shape[-2:] != (H, W):
            conf_4d = F.interpolate(
                conf_4d,
                size=(H, W),
                mode="bilinear",
                align_corners=False
            )

        if mono_4d is None:
            mono_4d = torch.zeros_like(depth_reg_4d)
        elif mono_4d.shape[-2:] != (H, W):
            mono_4d = F.interpolate(
                mono_4d,
                size=(H, W),
                mode="bilinear",
                align_corners=False
            )

        if edge is None:
            edge_4d = torch.zeros_like(depth_reg_4d)
        else:
            edge_4d = self._to_4d(edge)
            if edge_4d.shape[-2:] != (H, W):
                edge_4d = F.interpolate(
                    edge_4d,
                    size=(H, W),
                    mode="bilinear",
                    align_corners=False
                )

        x = torch.cat(
            [depth_reg_4d, conf_4d, mono_4d, edge_4d],
            dim=1
        )

        if self.refine_downsample is not None and self.refine_downsample > 1:
            h_small = max(1, H // self.refine_downsample)
            w_small = max(1, W // self.refine_downsample)

            x_small = F.interpolate(
                x,
                size=(h_small, w_small),
                mode="bilinear",
                align_corners=False
            )
        else:
            x_small = x

        feat = self.input_proj(x_small)

        local_feat = self.local_branch(feat)
        global_feat = self.global_attn(feat)

        fused_feat = torch.cat([local_feat, global_feat], dim=1)
        fused_feat = self.fuse(fused_feat)

        residual_small = self.out_conv(fused_feat)

        depth_residual = F.interpolate(
            residual_small,
            size=(H, W),
            mode="bilinear",
            align_corners=False
        )

        refined_depth = depth_reg_4d + self.residual_scale * depth_residual

        return refined_depth.squeeze(1), depth_residual.squeeze(1)


class SimpleGlobalAttention(nn.Module):
    """
    SGFormer-style simple global attention.

    It does not explicitly construct NxN attention.
    """

    def __init__(self, embed_dim=8):
        super(SimpleGlobalAttention, self).__init__()

        self.embed_dim = embed_dim

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, feat):
        B, C, H, W = feat.shape
        N = H * W

        tokens = feat.flatten(2).transpose(1, 2).contiguous()

        q = self.q_proj(tokens)
        k = self.k_proj(tokens)
        v = self.v_proj(tokens)

        kv = torch.bmm(k.transpose(1, 2), v) / float(N)
        out = torch.bmm(q, kv)

        out = self.out_proj(out)

        out = out.transpose(1, 2).reshape(B, C, H, W).contiguous()

        return out
