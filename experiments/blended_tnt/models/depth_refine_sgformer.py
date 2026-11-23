import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LightweightGlobalAttention(nn.Module):
    def __init__(self, dim=8, num_heads=1, downsample=16):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.downsample = downsample

        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.k_proj = nn.Linear(dim, dim, bias=True)
        self.v_proj = nn.Linear(dim, dim, bias=True)
        self.out_proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x):
        b, c, h, w = x.shape

        if self.downsample is not None and self.downsample > 1:
            h_ds = max(1, h // self.downsample)
            w_ds = max(1, w // self.downsample)
            x_small = F.interpolate(x, size=(h_ds, w_ds), mode="bilinear", align_corners=False)
        else:
            x_small = x
            h_ds, w_ds = h, w

        tokens = x_small.flatten(2).transpose(1, 2).contiguous()

        q = self.q_proj(tokens)
        k = self.k_proj(tokens)
        v = self.v_proj(tokens)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dim)
        attn = torch.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = self.out_proj(out)

        out = out.transpose(1, 2).contiguous().view(b, c, h_ds, w_ds)

        if h_ds != h or w_ds != w:
            out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)

        return out


class SGFormerDepthRefiner(nn.Module):
    def __init__(
        self,
        in_channels=4,
        embed_dim=8,
        num_heads=1,
        num_layers=1,
        residual_scale=0.05,
        refine_downsample=16,
    ):
        super().__init__()

        self.residual_scale = residual_scale

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

        self.global_attn = LightweightGlobalAttention(
            dim=embed_dim,
            num_heads=num_heads,
            downsample=refine_downsample,
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )

        hidden_dim = embed_dim
        self.out_conv = nn.Sequential(
            nn.Conv2d(embed_dim, hidden_dim, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, kernel_size=3, padding=1, bias=True),
        )

    def _to_4d(self, x, size=None):
        if x is None:
            return None

        if x.dim() == 3:
            x = x.unsqueeze(1)

        if size is not None and x.shape[-2:] != size:
            x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)

        return x

    def forward(self, depth_init, confidence=None, mono_depth=None, edge=None,
                depth_interval=None):
        depth_4d = self._to_4d(depth_init)
        size = depth_4d.shape[-2:]

        if confidence is None:
            conf_4d = torch.ones_like(depth_4d)
        else:
            conf_4d = self._to_4d(confidence, size=size)

        if mono_depth is None:
            mono_4d = depth_4d
        else:
            mono_4d = self._to_4d(mono_depth, size=size)

        if edge is None:
            edge_4d = torch.zeros_like(depth_4d)
        else:
            edge_4d = self._to_4d(edge, size=size)

        if depth_interval is None:
            interval_4d = torch.ones_like(depth_4d)
        else:
            interval_4d = self._to_4d(depth_interval, size=size).abs().clamp_min(1e-6)

        x = torch.cat([depth_4d, conf_4d, mono_4d, edge_4d], dim=1)

        guide_feat = self.input_proj(x)
        local_feat = self.local_branch(guide_feat)
        global_feat = self.global_attn(guide_feat)

        fuse_feat = self.fuse(torch.cat([local_feat, global_feat], dim=1))

        # Express the correction in local hypothesis-bin units.  Bounding and
        # confidence gating prevent a learned output bias from shifting an
        # entire depth map and damaging cross-view consistency.
        residual = torch.tanh(self.out_conv(fuse_feat))
        residual = residual * self.residual_scale * interval_4d
        residual = residual * conf_4d.detach().clamp(0.0, 1.0)
        refined_depth = depth_4d + residual

        return refined_depth.squeeze(1), residual.squeeze(1)
