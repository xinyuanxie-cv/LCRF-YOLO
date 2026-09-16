# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Block modules."""

from __future__ import annotations

from fileinput import close
from math import gamma

import torch
import torch.nn as nn
import torch.nn.functional as F
from networkx.classes.filters import hide_diedges
from numpy.ma.core import shape
from torch.ao.nn.quantized.functional import avg_pool2d
from triton.ops.blocksparse import softmax

# from PIL.ImageChops import offset
# from networkx.utils.misc import groups
# from numpy.ma.core import identity
# from pandas.io.clipboard import paste
# from sympy.physics.pring import energy
# from torch.cuda import device
# from torch.nn.functional import max_pool2d

from ultralytics.utils.torch_utils import fuse_conv_and_bn

from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, autopad
from .transformer import TransformerBlock

__all__ = (
    "C1",
    "C2",
    "C2PSA",
    "C3",
    "C3TR",
    "CIB",
    "DFL",
    "ELAN1",
    "PSA",
    "SPP",
    "SPPELAN",
    "SPPF",
    "AConv",
    "ADown",
    "Attention",
    "BNContrastiveHead",
    "Bottleneck",
    "BottleneckCSP",
    "C2f",
    "C2fAttn",
    "C2fCIB",
    "C2fPSA",
    "C3Ghost",
    "C3k2",
    "C3x",
    "CBFuse",
    "CBLinear",
    "ContrastiveHead",
    "GhostBottleneck",
    "HGBlock",
    "HGStem",
    "ImagePoolingAttn",
    "Proto",
    "RepC3",
    "RepNCSPELAN4",
    "RepVGGDW",
    "ResNetLayer",
    "SCDown",
    "TorchVision",
)

#from ...data.augment import Compose


class DFL(nn.Module):
    """Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1: int = 16):
        """Initialize a convolutional layer with a given number of input channels.

        Args:
            c1 (int): Number of input channels.
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the DFL module to input tensor and return transformed output."""
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """Ultralytics YOLO models mask Proto module for segmentation models."""

    def __init__(self, c1: int, c_: int = 256, c2: int = 32):
        """Initialize the Ultralytics YOLO models mask Proto module with specified number of protos and masks.

        Args:
            c1 (int): Input channels.
            c_ (int): Intermediate channels.
            c2 (int): Output channels (number of protos).
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1: int, cm: int, c2: int):
        """Initialize the StemBlock of PPHGNetV2.

        Args:
            c1 (int): Input channels.
            cm (int): Middle channels.
            c2 (int): Output channels.
        """
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(
        self,
        c1: int,
        cm: int,
        c2: int,
        k: int = 3,
        n: int = 6,
        lightconv: bool = False,
        shortcut: bool = False,
        act: nn.Module = nn.ReLU(),
    ):
        """Initialize HGBlock with specified parameters.

        Args:
            c1 (int): Input channels.
            cm (int): Middle channels.
            c2 (int): Output channels.
            k (int): Kernel size.
            n (int): Number of LightConv or Conv blocks.
            lightconv (bool): Whether to use LightConv.
            shortcut (bool): Whether to use shortcut connection.
            act (nn.Module): Activation function.
        """
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1: int, c2: int, k: tuple[int, ...] = (5, 9, 13)):
        """Initialize the SPP layer with input/output channels and pooling kernel sizes.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (tuple): Kernel sizes for max pooling.
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1: int, c2: int, k: int = 5, n: int = 3, shortcut: bool = False):
        """Initialize the SPPF layer with given input/output channels and kernel size.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size.
            n (int): Number of pooling iterations.
            shortcut (bool): Whether to use shortcut connection.

        Notes:
            This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1, act=False)
        self.cv2 = Conv(c_ * (n + 1), c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.n = n
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply sequential pooling operations to input and return concatenated feature maps."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(getattr(self, "n", 3)))
        y = self.cv2(torch.cat(y, 1))
        return y + x if getattr(self, "add", False) else y
#############
import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv


# class DASSPPF(nn.Module):
#     """
#     DASSPPF: Defect-Aware Selective SPPF
#
#     结构：
#     原始 SPPF 主分支
#     + 大核 DWConv 分支
#     + 条形卷积分支
#     + 局部对比分支
#     + Spatial Softmax Gate
#     + Strength Gate
#     + alpha_scale * tanh(alpha) 残差保护
#
#     注意：
#     alpha_scale 用于限制增强分支最大强度，防止 alpha 学到接近 1 后扰乱 P5 特征。
#     """
#
#     def __init__(
#         self,
#         c1,
#         c2,
#         k=5,
#         branch_k=7,
#         gate_ratio=4,
#         temperature=2.0,
#         use_strength=True,
#         alpha_init=0.05,
#         alpha_scale=0.10,
#         debug=True,
#         print_interval=500,
#     ):
#         super().__init__()
#
#         c_ = c1 // 2
#         hidden = max(c_ // gate_ratio, 16)
#
#         self.temperature = max(float(temperature), 1e-4)
#         self.use_strength = use_strength
#         self.alpha_scale = float(alpha_scale)
#
#         # debug 控制
#         self.debug = debug
#         self.debug_count = 0
#         self.print_interval = int(print_interval)
#
#         # -------------------------
#         # 1. 原始 SPPF 主分支
#         # -------------------------
#         self.cv1 = Conv(c1, c_, 1, 1)
#         self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
#         self.cv2 = Conv(c_ * 4, c2, 1, 1)
#
#         # -------------------------
#         # 2. 大核 DWConv 分支
#         # 适合区域型 / 块状 / 大范围纹理缺陷
#         # -------------------------
#         self.large_branch = nn.Sequential(
#             nn.Conv2d(
#                 c_,
#                 c_,
#                 kernel_size=branch_k,
#                 stride=1,
#                 padding=branch_k // 2,
#                 groups=c_,
#                 bias=False,
#             ),
#             nn.BatchNorm2d(c_),
#             nn.SiLU(inplace=True),
#             Conv(c_, c2, 1, 1),
#         )
#
#         # -------------------------
#         # 3. 条形卷积分支
#         # 适合 scratches / crazing / 细长裂纹结构
#         # -------------------------
#         self.strip_h = nn.Sequential(
#             nn.Conv2d(
#                 c_,
#                 c_,
#                 kernel_size=(1, branch_k),
#                 stride=1,
#                 padding=(0, branch_k // 2),
#                 groups=c_,
#                 bias=False,
#             ),
#             nn.BatchNorm2d(c_),
#             nn.SiLU(inplace=True),
#         )
#
#         self.strip_v = nn.Sequential(
#             nn.Conv2d(
#                 c_,
#                 c_,
#                 kernel_size=(branch_k, 1),
#                 stride=1,
#                 padding=(branch_k // 2, 0),
#                 groups=c_,
#                 bias=False,
#             ),
#             nn.BatchNorm2d(c_),
#             nn.SiLU(inplace=True),
#         )
#
#         self.strip_fuse = Conv(c_, c2, 1, 1)
#
#         # -------------------------
#         # 4. 局部对比分支
#         # 适合弱纹理 / 小缺陷 / 局部突变
#         # -------------------------
#         self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
#         self.local_branch = Conv(c_, c2, 1, 1)
#
#         # -------------------------
#         # 5. Spatial Softmax Gate
#         # 输出 [B, 3, H, W]
#         # 3 个通道分别对应 large / strip / local
#         # -------------------------
#         self.gate = nn.Sequential(
#             Conv(c_, hidden, 1, 1),
#             nn.Conv2d(hidden, 3, kernel_size=1, stride=1, padding=0, bias=True),
#         )
#
#         # -------------------------
#         # 6. Strength Gate
#         # 控制当前位置是否需要增强
#         # 输出 [B, 1, H, W]
#         # -------------------------
#         if self.use_strength:
#             self.strength_gate = nn.Sequential(
#                 Conv(c_, hidden, 1, 1),
#                 nn.Conv2d(hidden, 1, kernel_size=1, stride=1, padding=0, bias=True),
#                 nn.Sigmoid(),
#             )
#         else:
#             self.strength_gate = None
#
#         # -------------------------
#         # 7. 可学习残差强度
#         # 最终实际强度 = alpha_scale * tanh(alpha)
#         # -------------------------
#         self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
#
#     def forward(self, x):
#         # 先压缩通道
#         x = self.cv1(x)
#
#         # -------------------------
#         # 原始 SPPF 主分支
#         # -------------------------
#         y1 = self.m(x)
#         y2 = self.m(y1)
#         y3 = self.m(y2)
#         f_sppf = self.cv2(torch.cat((x, y1, y2, y3), dim=1))
#
#         # -------------------------
#         # 三个候选增强分支
#         # -------------------------
#         f_large = self.large_branch(x)
#
#         f_strip = self.strip_h(x) + self.strip_v(x)
#         f_strip = self.strip_fuse(f_strip)
#
#         f_lc = x - self.avg_pool(x)
#         f_lc = self.local_branch(f_lc)
#
#         # -------------------------
#         # Spatial Softmax Gate
#         # -------------------------
#         logits = self.gate(x) / self.temperature
#         weights = torch.softmax(logits, dim=1)
#
#         w_large = weights[:, 0:1, :, :]
#         w_strip = weights[:, 1:2, :, :]
#         w_lc = weights[:, 2:3, :, :]
#
#         # -------------------------
#         # 动态选择增强
#         # -------------------------
#         f_enh = w_large * f_large + w_strip * f_strip + w_lc * f_lc
#
#         # -------------------------
#         # Strength Gate
#         # -------------------------
#         if self.strength_gate is not None:
#             strength = self.strength_gate(x)
#             f_enh = strength * f_enh
#         else:
#             strength = None
#
#         # -------------------------
#         # alpha 上限保护
#         # 注意：这里最大增强幅度被限制为 alpha_scale
#         # 例如 alpha_scale=0.10，即使 tanh(alpha)=1，最大也只有 0.1
#         # -------------------------
#         alpha_tanh = torch.tanh(self.alpha)
#         alpha_eff = self.alpha_scale * alpha_tanh
#
#         # -------------------------
#         # Debug 打印
#         # 每 print_interval 次 forward 打印一次
#         # 只在训练阶段打印
#         # -------------------------
#         if self.debug and self.training:
#             if self.debug_count % self.print_interval == 0:
#                 strength_value = strength.mean().item() if strength is not None else -1.0
#
#                 print(
#                     f"[DASSPPF Debug] "
#                     f"w_large={w_large.mean().item():.4f}, "
#                     f"w_strip={w_strip.mean().item():.4f}, "
#                     f"w_lc={w_lc.mean().item():.4f}, "
#                     f"strength={strength_value:.4f}, "
#                     f"alpha_tanh={alpha_tanh.item():.4f}, "
#                     f"alpha_eff={alpha_eff.item():.4f}"
#                 )
#
#             self.debug_count += 1
#
#         # -------------------------
#         # 残差输出
#         # -------------------------
#         out = f_sppf + alpha_eff * f_enh
#
#         return out

import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv


class DASSPPF(nn.Module):
    """
    DASSPPF-C: Defect-Aware Selective SPPF - Compatible version

    设计目的：
    1. 保留原始 SPPF 主路径，保证稳定性；
    2. 只引入 large-kernel 分支和 strip 分支；
    3. 去掉 local contrast 分支，避免和 LCRB 功能重复；
    4. 去掉 strength gate，避免增强强度后期失控；
    5. 使用 alpha_scale 控制最大残差增强强度。

    输出：
    out = SPPF(x) + alpha_scale * tanh(alpha) * F_enh
    """

    def __init__(
        self,
        c1,
        c2,
        k=5,
        branch_k=7,
        gate_ratio=4,
        temperature=1.5,
        alpha_init=0.05,
        alpha_scale=0.05,
        debug=True,
        print_interval=500,
    ):
        super().__init__()

        # 保证 branch_k 是奇数
        branch_k = int(branch_k)
        if branch_k % 2 == 0:
            branch_k += 1

        c_ = c1 // 2
        hidden = max(c_ // gate_ratio, 16)

        self.temperature = max(float(temperature), 1e-4)
        self.alpha_scale = float(alpha_scale)

        self.debug = debug
        self.debug_count = 0
        self.print_interval = int(print_interval)

        # -------------------------------------------------
        # 1. Original SPPF branch
        # -------------------------------------------------
        self.cv1 = Conv(c1, c_, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)

        # -------------------------------------------------
        # 2. Large-kernel DWConv branch
        # 负责区域型 / 块状 / 大范围上下文
        # -------------------------------------------------
        self.large_branch = nn.Sequential(
            nn.Conv2d(
                c_,
                c_,
                kernel_size=branch_k,
                stride=1,
                padding=branch_k // 2,
                groups=c_,
                bias=False,
            ),
            nn.BatchNorm2d(c_),
            nn.SiLU(inplace=True),
            Conv(c_, c2, 1, 1),
        )

        # -------------------------------------------------
        # 3. Strip convolution branch
        # 负责细长型 / 方向性结构，如 scratches、crazing
        # -------------------------------------------------
        self.strip_h = nn.Sequential(
            nn.Conv2d(
                c_,
                c_,
                kernel_size=(1, branch_k),
                stride=1,
                padding=(0, branch_k // 2),
                groups=c_,
                bias=False,
            ),
            nn.BatchNorm2d(c_),
            nn.SiLU(inplace=True),
        )

        self.strip_v = nn.Sequential(
            nn.Conv2d(
                c_,
                c_,
                kernel_size=(branch_k, 1),
                stride=1,
                padding=(branch_k // 2, 0),
                groups=c_,
                bias=False,
            ),
            nn.BatchNorm2d(c_),
            nn.SiLU(inplace=True),
        )

        self.strip_fuse = Conv(c_, c2, 1, 1)

        # -------------------------------------------------
        # 4. Two-branch spatial gate
        # 输出 [B, 2, H, W]
        # 第 0 个通道控制 large 分支
        # 第 1 个通道控制 strip 分支
        # -------------------------------------------------
        self.gate = nn.Sequential(
            Conv(c_, hidden, 1, 1),
            nn.Conv2d(hidden, 2, kernel_size=1, stride=1, padding=0, bias=True),
        )

        # -------------------------------------------------
        # 5. Learnable residual scale
        # 最终有效增强强度 = alpha_scale * tanh(alpha)
        # -------------------------------------------------
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def forward(self, x):
        # 先压缩通道
        x = self.cv1(x)

        # -------------------------------------------------
        # Original SPPF
        # -------------------------------------------------
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        f_sppf = self.cv2(torch.cat((x, y1, y2, y3), dim=1))

        # -------------------------------------------------
        # Large branch
        # -------------------------------------------------
        f_large = self.large_branch(x)

        # -------------------------------------------------
        # Strip branch
        # -------------------------------------------------
        f_strip = self.strip_h(x) + self.strip_v(x)
        f_strip = self.strip_fuse(f_strip)

        # -------------------------------------------------
        # Spatial softmax gate
        # -------------------------------------------------
        logits = self.gate(x) / self.temperature
        weights = torch.softmax(logits, dim=1)

        w_large = weights[:, 0:1, :, :]
        w_strip = weights[:, 1:2, :, :]

        # -------------------------------------------------
        # Selective enhancement
        # -------------------------------------------------
        f_enh = w_large * f_large + w_strip * f_strip

        # -------------------------------------------------
        # Conservative residual
        # -------------------------------------------------
        alpha_tanh = torch.tanh(self.alpha)
        alpha_eff = self.alpha_scale * alpha_tanh

        if self.debug and self.training:
            if self.debug_count % self.print_interval == 0:
                print(
                    f"[DASSPPFC Debug] "
                    f"w_large={w_large.mean().item():.4f}, "
                    f"w_strip={w_strip.mean().item():.4f}, "
                    f"alpha_tanh={alpha_tanh.item():.4f}, "
                    f"alpha_eff={alpha_eff.item():.4f}"
                )
            self.debug_count += 1

        out = f_sppf + alpha_eff * f_enh

        return out
###############
import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv


class LQRC(nn.Module):
    """
    LQRC: Localization Quality-aware Residual Calibration

    作用：
    1. 放在 Detect 前，对检测尺度特征进行定位质量校准；
    2. 不做跨尺度融合，避免和 RCSFusion 重复；
    3. 不做局部对比增强，避免和 LCRB 重复；
    4. 通过空间质量权重 Q 引导局部残差校准；
    5. 使用 alpha_scale 控制最大残差强度，防止破坏原始检测特征。

    公式：
        Q = Sigmoid(Conv(DWConv(F)))
        R = PWConv(DWConv(F))
        F_out = F + alpha_scale * tanh(alpha) * Q * R
    """

    def __init__(
        self,
        c1,
        c2,
        k=3,
        alpha_init=0.05,
        alpha_scale=0.10,
        debug=True,
        print_interval=500,
    ):
        super().__init__()

        k = int(k)
        if k % 2 == 0:
            k += 1

        self.alpha_scale = float(alpha_scale)
        self.debug = debug
        self.debug_count = 0
        self.print_interval = int(print_interval)

        # 如果输入输出通道不同，先用 1x1 对齐
        self.proj = Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity()

        # 局部定位残差分支：只补充局部结构，不做复杂增强
        self.local_residual = nn.Sequential(
            nn.Conv2d(
                c2,
                c2,
                kernel_size=k,
                stride=1,
                padding=k // 2,
                groups=c2,
                bias=False,
            ),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
            Conv(c2, c2, 1, 1),
        )

        # 定位质量权重分支：输出 [B, 1, H, W]
        self.quality_gate = nn.Sequential(
            nn.Conv2d(
                c2,
                c2,
                kernel_size=k,
                stride=1,
                padding=k // 2,
                groups=c2,
                bias=False,
            ),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
            nn.Conv2d(c2, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid(),
        )

        # 可学习残差系数
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def forward(self, x):
        x = self.proj(x)

        q = self.quality_gate(x)
        r = self.local_residual(x)

        alpha_tanh = torch.tanh(self.alpha)
        alpha_eff = self.alpha_scale * alpha_tanh

        out = x + alpha_eff * q * r

        if self.debug and self.training:
            if self.debug_count % self.print_interval == 0:
                print(
                    f"[LQRC Debug] "
                    f"q_mean={q.mean().item():.4f}, "
                    f"q_min={q.min().item():.4f}, "
                    f"q_max={q.max().item():.4f}, "
                    f"alpha_tanh={alpha_tanh.item():.4f}, "
                    f"alpha_eff={alpha_eff.item():.4f}"
                )
            self.debug_count += 1

        return out

################

class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1: int, c2: int, n: int = 1):
        """Initialize the CSP Bottleneck with 1 convolution.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of convolutions.
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution and residual connection to input tensor."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize a CSP Bottleneck with 2 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        """Initialize a CSP bottleneck with 2 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize the CSP Bottleneck with 3 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CSP bottleneck with 3 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize C3 module with cross-convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1: int, c2: int, n: int = 3, e: float = 1.0):
        """Initialize RepC3 module with RepConv blocks.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of RepConv blocks.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of RepC3 module."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize C3 module with TransformerBlock.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Transformer blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize C3 module with GhostBottleneck.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Ghost bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/Efficient-AI-Backbones."""

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1):
        """Initialize Ghost Bottleneck module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size.
            s (int): Stride.
        """
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False),  # pw-linear
        )
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply skip connection and addition to input tensor."""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5
    ):
        """Initialize a standard bottleneck module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            g (int): Groups for convolutions.
            k (tuple): Kernel sizes for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply bottleneck with optional shortcut connection."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize CSP Bottleneck.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply CSP bottleneck with 4 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class ResNetBlock(nn.Module):
    """ResNet block with standard convolution layers."""

    def __init__(self, c1: int, c2: int, s: int = 1, e: int = 4):
        """Initialize ResNet block.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            s (int): Stride.
            e (int): Expansion ratio.
        """
        super().__init__()
        c3 = e * c2
        self.cv1 = Conv(c1, c2, k=1, s=1, act=True)
        self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True)
        self.cv3 = Conv(c2, c3, k=1, act=False)
        self.shortcut = nn.Sequential(Conv(c1, c3, k=1, s=s, act=False)) if s != 1 or c1 != c3 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the ResNet block."""
        return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))


class ResNetLayer(nn.Module):
    """ResNet layer with multiple ResNet blocks."""

    def __init__(self, c1: int, c2: int, s: int = 1, is_first: bool = False, n: int = 1, e: int = 4):
        """Initialize ResNet layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            s (int): Stride.
            is_first (bool): Whether this is the first layer.
            n (int): Number of ResNet blocks.
            e (int): Expansion ratio.
        """
        super().__init__()
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(
                Conv(c1, c2, k=7, s=2, p=3, act=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            blocks = [ResNetBlock(c1, c2, s, e=e)]
            blocks.extend([ResNetBlock(e * c2, c2, 1, e=e) for _ in range(n - 1)])
            self.layer = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the ResNet layer."""
        return self.layer(x)


class MaxSigmoidAttnBlock(nn.Module):
    """Max Sigmoid attention block."""

    def __init__(self, c1: int, c2: int, nh: int = 1, ec: int = 128, gc: int = 512, scale: bool = False):
        """Initialize MaxSigmoidAttnBlock.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            nh (int): Number of heads.
            ec (int): Embedding channels.
            gc (int): Guide channels.
            scale (bool): Whether to use learnable scale parameter.
        """
        super().__init__()
        self.nh = nh
        self.hc = c2 // nh
        self.ec = Conv(c1, ec, k=1, act=False) if c1 != ec else None
        self.gl = nn.Linear(gc, ec)
        self.bias = nn.Parameter(torch.zeros(nh))
        self.proj_conv = Conv(c1, c2, k=3, s=1, act=False)
        self.scale = nn.Parameter(torch.ones(1, nh, 1, 1)) if scale else 1.0

    def forward(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        """Forward pass of MaxSigmoidAttnBlock.

        Args:
            x (torch.Tensor): Input tensor.
            guide (torch.Tensor): Guide tensor.

        Returns:
            (torch.Tensor): Output tensor after attention.
        """
        bs, _, h, w = x.shape

        guide = self.gl(guide)
        guide = guide.view(bs, guide.shape[1], self.nh, self.hc)
        embed = self.ec(x) if self.ec is not None else x
        embed = embed.view(bs, self.nh, self.hc, h, w)

        aw = torch.einsum("bmchw,bnmc->bmhwn", embed, guide)
        aw = aw.max(dim=-1)[0]
        aw = aw / (self.hc**0.5)
        aw = aw + self.bias[None, :, None, None]
        aw = aw.sigmoid() * self.scale

        x = self.proj_conv(x)
        x = x.view(bs, self.nh, -1, h, w)
        x = x * aw.unsqueeze(2)
        return x.view(bs, -1, h, w)


class C2fAttn(nn.Module):
    """C2f module with an additional attn module."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        ec: int = 128,
        nh: int = 1,
        gc: int = 512,
        shortcut: bool = False,
        g: int = 1,
        e: float = 0.5,
    ):
        """Initialize C2f module with attention mechanism.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            ec (int): Embedding channels for attention.
            nh (int): Number of heads for attention.
            gc (int): Guide channels for attention.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((3 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.attn = MaxSigmoidAttnBlock(self.c, self.c, gc=gc, ec=ec, nh=nh)

    def forward(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        """Forward pass through C2f layer with attention.

        Args:
            x (torch.Tensor): Input tensor.
            guide (torch.Tensor): Guide tensor for attention.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        """Forward pass using split() instead of chunk().

        Args:
            x (torch.Tensor): Input tensor.
            guide (torch.Tensor): Guide tensor for attention.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))


class ImagePoolingAttn(nn.Module):
    """ImagePoolingAttn: Enhance the text embeddings with image-aware information."""

    def __init__(
        self, ec: int = 256, ch: tuple[int, ...] = (), ct: int = 512, nh: int = 8, k: int = 3, scale: bool = False
    ):
        """Initialize ImagePoolingAttn module.

        Args:
            ec (int): Embedding channels.
            ch (tuple): Channel dimensions for feature maps.
            ct (int): Channel dimension for text embeddings.
            nh (int): Number of attention heads.
            k (int): Kernel size for pooling.
            scale (bool): Whether to use learnable scale parameter.
        """
        super().__init__()

        nf = len(ch)
        self.query = nn.Sequential(nn.LayerNorm(ct), nn.Linear(ct, ec))
        self.key = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.value = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.proj = nn.Linear(ec, ct)
        self.scale = nn.Parameter(torch.tensor([0.0]), requires_grad=True) if scale else 1.0
        self.projections = nn.ModuleList([nn.Conv2d(in_channels, ec, kernel_size=1) for in_channels in ch])
        self.im_pools = nn.ModuleList([nn.AdaptiveMaxPool2d((k, k)) for _ in range(nf)])
        self.ec = ec
        self.nh = nh
        self.nf = nf
        self.hc = ec // nh
        self.k = k

    def forward(self, x: list[torch.Tensor], text: torch.Tensor) -> torch.Tensor:
        """Forward pass of ImagePoolingAttn.

        Args:
            x (list[torch.Tensor]): List of input feature maps.
            text (torch.Tensor): Text embeddings.

        Returns:
            (torch.Tensor): Enhanced text embeddings.
        """
        bs = x[0].shape[0]
        assert len(x) == self.nf
        num_patches = self.k**2
        x = [pool(proj(x)).view(bs, -1, num_patches) for (x, proj, pool) in zip(x, self.projections, self.im_pools)]
        x = torch.cat(x, dim=-1).transpose(1, 2)
        q = self.query(text)
        k = self.key(x)
        v = self.value(x)

        # q = q.reshape(1, text.shape[1], self.nh, self.hc).repeat(bs, 1, 1, 1)
        q = q.reshape(bs, -1, self.nh, self.hc)
        k = k.reshape(bs, -1, self.nh, self.hc)
        v = v.reshape(bs, -1, self.nh, self.hc)

        aw = torch.einsum("bnmc,bkmc->bmnk", q, k)
        aw = aw / (self.hc**0.5)
        aw = F.softmax(aw, dim=-1)

        x = torch.einsum("bmnk,bkmc->bnmc", aw, v)
        x = self.proj(x.reshape(bs, -1, self.ec))
        return x * self.scale + text


class ContrastiveHead(nn.Module):
    """Implements contrastive learning head for region-text similarity in vision-language models."""

    def __init__(self):
        """Initialize ContrastiveHead with region-text similarity parameters."""
        super().__init__()
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.07).log())

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Forward function of contrastive learning.

        Args:
            x (torch.Tensor): Image features.
            w (torch.Tensor): Text features.

        Returns:
            (torch.Tensor): Similarity scores.
        """
        x = F.normalize(x, dim=1, p=2)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class BNContrastiveHead(nn.Module):
    """Batch Norm Contrastive Head using batch norm instead of l2-normalization.

    Args:
        embed_dims (int): Embed dimensions of text and image features.
    """

    def __init__(self, embed_dims: int):
        """Initialize BNContrastiveHead.

        Args:
            embed_dims (int): Embedding dimensions for features.
        """
        super().__init__()
        self.norm = nn.BatchNorm2d(embed_dims)
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        # use -1.0 is more stable
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))

    def fuse(self):
        """Fuse the batch normalization layer in the BNContrastiveHead module."""
        del self.norm
        del self.bias
        del self.logit_scale
        self.forward = self.forward_fuse

    @staticmethod
    def forward_fuse(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Passes image features through unchanged after fusing."""
        return x

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Forward function of contrastive learning with batch normalization.

        Args:
            x (torch.Tensor): Image features.
            w (torch.Tensor): Text features.

        Returns:
            (torch.Tensor): Similarity scores.
        """
        x = self.norm(x)
        w = F.normalize(w, dim=-1, p=2)

        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class RepBottleneck(Bottleneck):
    """Rep bottleneck."""

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5
    ):
        """Initialize RepBottleneck.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            g (int): Groups for convolutions.
            k (tuple): Kernel sizes for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RepConv(c1, c_, k[0], 1)


class RepCSP(C3):
    """Repeatable Cross Stage Partial Network (RepCSP) module for efficient feature extraction."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize RepCSP layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of RepBottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


class RepNCSPELAN4(nn.Module):
    """CSP-ELAN."""

    def __init__(self, c1: int, c2: int, c3: int, c4: int, n: int = 1):
        """Initialize CSP-ELAN layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            c3 (int): Intermediate channels.
            c4 (int): Intermediate channels for RepCSP.
            n (int): Number of RepCSP blocks.
        """
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepCSP(c3 // 2, c4, n), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepCSP(c4, c4, n), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through RepNCSPELAN4 layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class ELAN1(RepNCSPELAN4):
    """ELAN1 module with 4 convolutions."""

    def __init__(self, c1: int, c2: int, c3: int, c4: int):
        """Initialize ELAN1 layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            c3 (int): Intermediate channels.
            c4 (int): Intermediate channels for convolutions.
        """
        super().__init__(c1, c2, c3, c4)
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = Conv(c3 // 2, c4, 3, 1)
        self.cv3 = Conv(c4, c4, 3, 1)
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)


class AConv(nn.Module):
    """AConv."""

    def __init__(self, c1: int, c2: int):
        """Initialize AConv module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, 3, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through AConv layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        return self.cv1(x)


class ADown(nn.Module):
    """ADown."""

    def __init__(self, c1: int, c2: int):
        """Initialize ADown module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
        """
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ADown layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)


class SPPELAN(nn.Module):
    """SPP-ELAN."""

    def __init__(self, c1: int, c2: int, c3: int, k: int = 5):
        """Initialize SPP-ELAN block.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            c3 (int): Intermediate channels.
            k (int): Kernel size for max pooling.
        """
        super().__init__()
        self.c = c3
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv4 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c3, c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through SPPELAN layer."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        return self.cv5(torch.cat(y, 1))


class CBLinear(nn.Module):
    """CBLinear."""

    def __init__(self, c1: int, c2s: list[int], k: int = 1, s: int = 1, p: int | None = None, g: int = 1):
        """Initialize CBLinear module.

        Args:
            c1 (int): Input channels.
            c2s (list[int]): List of output channel sizes.
            k (int): Kernel size.
            s (int): Stride.
            p (int | None): Padding.
            g (int): Groups.
        """
        super().__init__()
        self.c2s = c2s
        self.conv = nn.Conv2d(c1, sum(c2s), k, s, autopad(k, p), groups=g, bias=True)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Forward pass through CBLinear layer."""
        return self.conv(x).split(self.c2s, dim=1)


class CBFuse(nn.Module):
    """CBFuse."""
class SDC3k2(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True):
        super().__init__()

        # 两个分支
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c1, c2, 1, 1)

        # 深层分支（Bottleneck堆）
        self.m = nn.Sequential(
            *[Bottleneck(c2, c2, shortcut) for _ in range(n)]
        )

        # ⭐ Spatial Attention（核心创新）
    def __init__(self, idx: list[int]):
        """Initialize CBFuse module.

        Args:
            idx (list[int]): Indices for feature selection.
        """
        super().__init__()
        self.idx = idx

    def forward(self, xs: list[torch.Tensor]) -> torch.Tensor:
        """Forward pass through CBFuse layer.

        Args:
            xs (list[torch.Tensor]): List of input tensors.

        Returns:
            (torch.Tensor): Fused output tensor.
        """
        target_size = xs[-1].shape[2:]
        res = [F.interpolate(x[self.idx[i]], size=target_size, mode="nearest") for i, x in enumerate(xs[:-1])]
        return torch.sum(torch.stack(res + xs[-1:]), dim=0)


class C3f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        """Initialize CSP bottleneck layer with three convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv((2 + n) * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(c_, c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through C3f layer."""
        y = [self.cv2(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv3(torch.cat(y, 1))


class C3k2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
    ):
        """Initialize C3k2 module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of blocks.
            c3k (bool): Whether to use C3k blocks.
            e (float): Expansion ratio.
            attn (bool): Whether to use attention blocks.
            g (int): Groups for convolutions.
            shortcut (bool): Whether to use shortcut connections.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            nn.Sequential(
                Bottleneck(self.c, self.c, shortcut, g),
                PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1)),
            )
            if attn
            else C3k(self.c, self.c, 2, shortcut, g)
            if c3k
            else Bottleneck(self.c, self.c, shortcut, g)
            for _ in range(n)
        )

#####有涨进的ADC3K2
class ADGuide(nn.Module):
    """
    Anisotropic Diffusion Guide
    思路：
    - 平滑区域：更强扩散
    - 边界区域：抑制扩散
    - 用可学习方式近似各向异性扩散
    """

    def __init__(self, c: int, k: int = 3):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=k, stride=1, padding=k // 2, groups=c, bias=False),
            nn.BatchNorm2d(c),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 局部平滑项
        smooth = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)

        # 梯度近似：边界/异常区域响应会更大
        grad = torch.abs(x - smooth)

        # 自适应尺度，避免不同通道数值范围差异太大
        scale = grad.mean(dim=(2, 3), keepdim=True).detach() + 1e-6

        # 各向异性扩散系数：平滑区域高，边界区域低
        conduct = torch.exp(-((grad / scale) ** 2))

        # 再做一次轻量细化，让它可学习
        conduct = self.refine(conduct)

        # 扩散更新：平滑区域更接近 smooth，边界区域更保留原值
        out = x + conduct * (smooth - x)
        return out


class ADBottleneck(nn.Module):
    """
    AD Bottleneck
    设计目标：
    1. 保留 cv1 / cv2 命名，尽量兼容官方预训练
    2. 在中间特征上引入各向异性扩散引导
    3. alpha=0 初始化，训练初期尽量接近原始 Bottleneck
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        shortcut: bool = True,
        g: int = 1,
        e: float = 1.0,
        c3k: bool = False,
    ):
        super().__init__()
        c_ = int(c2 * e)

        self.cv1 = Conv(c1, c_, 3, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)

        self.ad = ADGuide(c_)

        # 初始为 0，保证一开始更像原始 Bottleneck
        self.alpha = nn.Parameter(torch.zeros(1))
        #self.alphe = nn.Parameter(torch.tensor(0.1))   ######把他的alpha改进从1到0.1试一下
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv1(x)

        y_ad = self.ad(y)
        y = y + self.alpha * (y_ad - y)

        y = self.cv2(y)
        return x + y if self.add else y


class ADC3k2(C2f):
    """
    Anisotropic-Diffusion C3k2
    接口保持和 YOLO11 的 C3k2 一致，可直接在 YAML 中替换
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
    ):
        super().__init__(c1, c2, n, shortcut, g, e)

        self.m = nn.ModuleList(
            [
                ADBottleneck(
                    self.c,
                    self.c,
                    shortcut=shortcut,
                    g=g,
                    e=1.0,
                    c3k=c3k,
                )
                for _ in range(n)
            ]
        )

        self.use_attn = attn
        if attn:
            self.out_attn = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(c2, c2, kernel_size=1, stride=1, padding=0, bias=True),
                nn.Sigmoid(),
            )
            self.out_alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        out = self.cv2(torch.cat(y, 1))

        if self.use_attn:
            out = out * (1.0 + self.out_alpha * self.out_attn(out))

        return out


# class MSEFBlock(nn.Module):
#     """
#     轻量 MSEF：
#     - DWConv 做局部空间细化
#     - SE 做通道自适应重标定
#     - 残差式增强，更稳
#     """
#     def __init__(self, ch: int, reduction_ratio: int = 8):
#         super().__init__()
#         hidden = max(ch // reduction_ratio, 8)
#
#         self.dw = nn.Sequential(
#             nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, groups=ch, bias=False),
#             nn.BatchNorm2d(ch),
#             nn.SiLU(),
#         )
#
#         self.se = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(ch, hidden, kernel_size=1, bias=False),
#             nn.SiLU(),
#             nn.Conv2d(hidden, ch, kernel_size=1, bias=True),
#             nn.Sigmoid(),
#         )
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         y = self.dw(x)
#         y = y * self.se(y)
#         return x + y
#
#
# class MSEBottleneck(nn.Module):
#     """
#     只融入 MSEF 的 Bottleneck
#     - 保留 cv1 / cv2，尽量兼容预训练
#     - beta=0 初始化，训练初期更接近原始 Bottleneck
#     """
#     def __init__(
#         self,
#         c1: int,
#         c2: int,
#         shortcut: bool = True,
#         g: int = 1,
#         e: float = 1.0,
#         c3k: bool = False,
#     ):
#         super().__init__()
#         c_ = int(c2 * e)
#
#         self.cv1 = Conv(c1, c_, 3, 1)
#         self.cv2 = Conv(c_, c2, 3, 1, g=g)
#
#         self.msef = MSEFBlock(c_)
#         self.beta = nn.Parameter(torch.zeros(1))
#
#         self.add = shortcut and c1 == c2
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         y = self.cv1(x)
#
#         # MSEF 细化
#         y_msef = self.msef(y)
#
#         # 残差渐进融合，初期更稳
#         y = y + self.beta * (y_msef - y)
#
#         y = self.cv2(y)
#         return x + y if self.add else y
#
#
# class ADC3k2(C2f):  # Real name: MSEC3k2
#     """
#     纯 MSEF 融合版 C3k2
#     类名保持 ADC3k2 不变，方便你直接复用原来的 yaml / tasks 注册
#     """
#     def __init__(
#         self,
#         c1: int,
#         c2: int,
#         n: int = 1,
#         c3k: bool = False,
#         e: float = 0.5,
#         attn: bool = False,
#         g: int = 1,
#         shortcut: bool = True,
#     ):
#         super().__init__(c1, c2, n, shortcut, g, e)
#
#         self.m = nn.ModuleList(
#             [
#                 MSEBottleneck(
#                     self.c,
#                     self.c,
#                     shortcut=shortcut,
#                     g=g,
#                     e=1.0,
#                     c3k=c3k,
#                 )
#                 for _ in range(n)
#             ]
#         )
#
#         self.use_attn = attn
#         if attn:
#             self.out_attn = nn.Sequential(
#                 nn.AdaptiveAvgPool2d(1),
#                 nn.Conv2d(c2, c2, kernel_size=1, bias=True),
#                 nn.Sigmoid(),
#             )
#             self.out_alpha = nn.Parameter(torch.zeros(1))
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         y = list(self.cv1(x).chunk(2, 1))
#         y.extend(m(y[-1]) for m in self.m)
#         out = self.cv2(torch.cat(y, 1))
#
#         if self.use_attn:
#             out = out * (1.0 + self.out_alpha * self.out_attn(out))
#
#         return out
########

#xinmokuai

#########
class QRC_SimAM(nn.Module):
    """
    Parameter-free SimAM-style attention.
    It enhances informative responses without introducing extra learnable parameters.
    """

    def __init__(self, e_lambda=1e-4):
        super().__init__()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        n = h * w - 1

        mean = x.mean(dim=(2, 3), keepdim=True)
        var = ((x - mean) ** 2).sum(dim=(2, 3), keepdim=True) / (n + self.e_lambda)

        energy = (x - mean) ** 2 / (4 * (var + self.e_lambda)) + 0.5
        return x * torch.sigmoid(energy)


def _make_gn(c, max_groups=8):
    """
    Make GroupNorm safely for different channel numbers.
    """
    for g in [max_groups, 4, 2, 1]:
        if c % g == 0:
            return nn.GroupNorm(g, c)
    return nn.GroupNorm(1, c)


class QRCCalib(nn.Module):
    """
    QRCCalib: Quality-aware Residual Calibration before Detect.

    This module is designed to be placed before Detect on P3/P4/P5.
    It performs a very weak residual calibration with alpha initialized to 0,
    so the module is identity at the beginning and will not disturb the trained
    feature distribution aggressively.

    Recommended usage:
        P3 -> QRCCalib -> Detect
        P4 -> QRCCalib -> Detect
        P5 -> QRCCalib -> Detect
    """

    def __init__(self, c1, c2, scale=0.05):
        super().__init__()

        self.proj = nn.Identity() if c1 == c2 else nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=1, stride=1, padding=0, bias=False),
            _make_gn(c2),
            nn.SiLU()
        )

        self.local_refine = nn.Sequential(
            nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1, groups=c2, bias=False),
            _make_gn(c2),
            nn.SiLU(),
            nn.Conv2d(c2, c2, kernel_size=1, stride=1, padding=0, bias=False),
            _make_gn(c2),
            nn.SiLU()
        )

        self.simam = QRC_SimAM()

        self.quality_gate = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.SiLU(),
            nn.Conv2d(16, 1, kernel_size=7, stride=1, padding=3, bias=True),
            nn.Sigmoid()
        )

        self.alpha = nn.Parameter(torch.zeros(1))
        self.scale = scale

    def forward(self, x):
        x = self.proj(x)

        avg_x = torch.mean(x, dim=1, keepdim=True)
        max_x, _ = torch.max(x, dim=1, keepdim=True)
        std_x = torch.std(x, dim=1, keepdim=True, unbiased=False)

        smooth_x = F.avg_pool2d(
            avg_x,
            kernel_size=5,
            stride=1,
            padding=2,
            count_include_pad=False
        )
        local_contrast = torch.abs(avg_x - smooth_x)

        gate_in = torch.cat([avg_x, max_x, std_x, local_contrast], dim=1)
        q_gate = self.quality_gate(gate_in)

        detail = self.local_refine(x)
        detail = self.simam(detail)

        gamma = self.scale * torch.tanh(self.alpha)

        return x + gamma * q_gate * detail

#####新加的c3k2模块
class DirectionalTextureResidual(nn.Module):
    """
    TRC3k2-v1 stable: Directional Texture Residual Branch
    方向纹理残差分支

    目的：
    - 捕获钢材表面缺陷中的方向性纹理扰动
    - 适合 scratches / crazing / rolled-in_scale / pitted_surface 等纹理型缺陷
    - 保持轻量化，不使用 Transformer / 大卷积 / 通用注意力
    """

    def __init__(self, c: int, k: int = 7):
        super().__init__()
        assert k % 2 == 1, "DirectionalTextureResidual kernel size should be odd."
        self.k = k

        # 水平/垂直方向残差融合
        self.fuse = nn.Sequential(
            nn.Conv2d(2 * c, c, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(),
            nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, groups=c, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(),
        )

        # 轻量空间 gate，用于控制纹理残差注入位置
        self.gate = nn.Sequential(
            nn.Conv2d(4, 1, kernel_size=7, stride=1, padding=3, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        # 水平方向平滑：突出竖向/斜向纹理扰动
        smooth_h = F.avg_pool2d(
            x,
            kernel_size=(1, self.k),
            stride=1,
            padding=(0, self.k // 2),
            count_include_pad=False,
        )

        # 垂直方向平滑：突出横向/斜向纹理扰动
        smooth_v = F.avg_pool2d(
            x,
            kernel_size=(self.k, 1),
            stride=1,
            padding=(self.k // 2, 0),
            count_include_pad=False,
        )

        # 方向纹理残差
        r_h = x - smooth_h
        r_v = x - smooth_v

        # 融合方向残差
        tex = self.fuse(torch.cat([r_h, r_v], dim=1))

        # 残差强度图
        mag = torch.abs(r_h) + torch.abs(r_v)

        avg_mag = torch.mean(mag, dim=1, keepdim=True)
        max_mag, _ = torch.max(mag, dim=1, keepdim=True)
        avg_x = torch.mean(x, dim=1, keepdim=True)
        std_x = torch.std(x, dim=1, keepdim=True, unbiased=False)

        g = self.gate(torch.cat([avg_mag, max_mag, avg_x, std_x], dim=1))

        return tex, g


class TRCBottleneck(nn.Module):
    """
    TRC3k2-v1 stable Bottleneck

    注意：
    - 这是唯一稳定版
    - scale=0.5
    - c3k=True 时 k_tex=9
    - 不加入 dominance，不降 scale
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        shortcut: bool = True,
        g: int = 1,
        e: float = 1.0,
        c3k: bool = False,
        scale: float = 0.1,
    ):
        super().__init__()
        c_ = int(c2 * e)

        self.cv1 = Conv(c1, c_, 3, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)

        # v1 原始设置：c3k=True 时扩大方向纹理感受野
        k_tex =  7

        self.tr = DirectionalTextureResidual(c_, k=k_tex)

        # alpha=0 初始化，训练初期接近原始 Bottleneck
        self.alpha = nn.Parameter(torch.zeros(1))
        self.scale = scale

        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv1(x)

        tex, g = self.tr(y)

        gamma = self.scale * torch.tanh(self.alpha)

        # 方向纹理残差注入
        y = y + gamma * g * tex

        y = self.cv2(y)

        return x + y if self.add else y


class TRC3k2(C2f):
    """
    TRC3k2-v1 stable

    Texture-Ridge Continuity C3k2
    面向钢材表面缺陷检测的方向纹理连续性 C3k2。
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
    ):
        super().__init__(c1, c2, n, shortcut, g, e)

        self.m = nn.ModuleList(
            [
                TRCBottleneck(
                    self.c,
                    self.c,
                    shortcut=shortcut,
                    g=g,
                    e=1.0,
                    c3k=c3k,
                    scale=0.1,
                )
                for _ in range(n)
            ]
        )

        # 保留接口，当前稳定版不启用额外注意力
        self.use_attn = attn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        out = self.cv2(torch.cat(y, 1))
        return out
############
class MBAContextBranch(nn.Module):
    """
    Base context branch.

    This branch preserves stable contextual representation and prevents
    the module from over-focusing on a single defect morphology.
    """

    def __init__(self, c: int, n: int = 1):
        super().__init__()

        self.blocks = nn.ModuleList([
            nn.Sequential(
                Conv(c, c, 1, 1),
                Conv(c, c, 3, 1, g=c),
                Conv(c, c, 1, 1),
            )
            for _ in range(n)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x
        for block in self.blocks:
            y = y + block(y)
        return y


class MBADirectionBranch(nn.Module):
    """
    Directional morphology branch.

    This branch extracts horizontal/vertical directional residuals,
    which are useful for direction-continuous defects such as scratches,
    rolled-in_scale, and patches.
    """

    def __init__(self, c: int, k: int = 7):
        super().__init__()
        assert k % 2 == 1, "MBA directional kernel size should be odd."
        self.k = k

        self.fuse = nn.Sequential(
            Conv(2 * c, c, 1, 1),
            Conv(c, c, 3, 1, g=c),
        )

    def forward(self, x: torch.Tensor):
        smooth_h = F.avg_pool2d(
            x,
            kernel_size=(1, self.k),
            stride=1,
            padding=(0, self.k // 2),
            count_include_pad=False,
        )

        smooth_v = F.avg_pool2d(
            x,
            kernel_size=(self.k, 1),
            stride=1,
            padding=(self.k // 2, 0),
            count_include_pad=False,
        )

        r_h = x - smooth_h
        r_v = x - smooth_v

        feat = self.fuse(torch.cat([r_h, r_v], dim=1))

        mag = torch.abs(r_h) + torch.abs(r_v)
        mag = torch.mean(mag, dim=1, keepdim=True)

        return feat, mag


class MBALocalWeakBranch(nn.Module):
    """
    Local weak-texture branch.

    This branch captures local low-contrast and fragmented responses,
    which helps protect weak/discrete defects such as crazing and inclusion.
    """

    def __init__(self, c: int):
        super().__init__()

        self.fuse = nn.Sequential(
            Conv(2 * c, c, 1, 1),
            Conv(c, c, 3, 1, g=c),
        )

        self.refine = Conv(c, c, 1, 1)

    def forward(self, x: torch.Tensor):
        local_3 = x - F.avg_pool2d(
            x,
            kernel_size=3,
            stride=1,
            padding=1,
            count_include_pad=False,
        )

        local_5 = x - F.avg_pool2d(
            x,
            kernel_size=5,
            stride=1,
            padding=2,
            count_include_pad=False,
        )

        feat = self.fuse(torch.cat([local_3, local_5], dim=1))
        feat = self.refine(feat)

        mag = torch.abs(local_3) + torch.abs(local_5)
        mag = torch.mean(mag, dim=1, keepdim=True)

        return feat, mag


class MBAMorphologyGate(nn.Module):
    """
    Morphology-adaptive gate.

    This gate performs branch-level morphology selection among:
    - base context branch
    - directional morphology branch
    - local weak-texture branch

    Softmax is used so that the three morphology branches compete and
    adaptively share the contribution at each spatial position.
    """

    def __init__(self):
        super().__init__()

        self.gate = nn.Sequential(
            nn.Conv2d(5, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.SiLU(),
            nn.Conv2d(16, 3, kernel_size=7, stride=1, padding=3, bias=True),
        )

    def forward(
        self,
        x: torch.Tensor,
        dir_mag: torch.Tensor,
        loc_mag: torch.Tensor,
    ):
        avg_x = torch.mean(x, dim=1, keepdim=True)
        max_x, _ = torch.max(x, dim=1, keepdim=True)
        std_x = torch.std(x, dim=1, keepdim=True, unbiased=False)

        gate_in = torch.cat(
            [
                dir_mag,
                loc_mag,
                avg_x,
                max_x,
                std_x,
            ],
            dim=1,
        )

        weight = self.gate(gate_in)
        weight = torch.softmax(weight, dim=1)

        w_base = weight[:, 0:1, :, :]
        w_dir = weight[:, 1:2, :, :]
        w_loc = weight[:, 2:3, :, :]

        return w_base, w_dir, w_loc


class MBABlock(nn.Module):
    """
    MBA-Block: Morphology-Balanced Aggregation Block.

    This is the original replacement-style MBA-Block.

    Core idea:
    - Base branch preserves stable contextual representation.
    - Direction branch models direction-continuous defects.
    - Local weak branch protects low-contrast/discrete defects.
    - Morphology gate adaptively selects among different morphology branches.

    Recommended setting:
        Replace the third backbone C3k2 / layer6.
        res_scale = 0.5.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
        res_scale: float = 0.5,
    ):
        super().__init__()

        c_ = int(c2 * e)

        self.cv_in = Conv(c1, c_, 1, 1)

        self.base_branch = MBAContextBranch(c_, n=max(n, 1))

        # Original MBA setting.
        k_dir = 7 if c3k else 5
        self.dir_branch = MBADirectionBranch(c_, k=k_dir)

        self.loc_branch = MBALocalWeakBranch(c_)

        self.morph_gate = MBAMorphologyGate()

        self.fuse = Conv(4 * c_, c2, 1, 1)

        self.use_shortcut = shortcut and c1 == c2
        self.res_scale = res_scale

        # Keep interface compatibility.
        self.use_attn = attn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = self.cv_in(x)

        f_base = self.base_branch(x_in)

        f_dir, dir_mag = self.dir_branch(f_base)
        f_loc, loc_mag = self.loc_branch(f_base)

        w_base, w_dir, w_loc = self.morph_gate(f_base, dir_mag, loc_mag)

        f_balanced = w_base * f_base + w_dir * f_dir + w_loc * f_loc

        out = self.fuse(
            torch.cat(
                [
                    f_base,
                    w_dir * f_dir,
                    w_loc * f_loc,
                    f_balanced,
                ],
                dim=1,
            )
        )

        if self.use_shortcut:
            out = x + self.res_scale * out

        return out
################
class IAN_Stem(nn.Module):
    """
    IAN-Stem: Information-Aware dual-branch stem.

    This module replaces the first stride-2 Conv in YOLO11n.
    The main branch performs standard convolutional downsampling,
    while the auxiliary pooling branch preserves stronger local responses
    during early downsampling to reduce the loss of weak defect cues.
    """

    def __init__(self, c1, c2):
        super().__init__()

        c_ = c2 // 2

        # Main branch: standard convolutional downsampling
        self.branch_main = Conv(c1, c_, k=3, s=2)

        # Auxiliary branch: pooling-based detail-preserving downsampling
        self.branch_aux = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            Conv(c1, c_, k=1, s=1)
        )

        # Merge two branches
        self.merge = Conv(c2, c2, k=1, s=1)

    def forward(self, x):
        x_main = self.branch_main(x)
        x_aux = self.branch_aux(x)

        x = torch.cat((x_main, x_aux), dim=1)
        x = self.merge(x)

        return x
###########
import torch
import torch.nn as nn


# ==============================================================================
# 1. 浅层高频保真器 (High-Frequency Detail Extractor)
# 科学目的：针对浅层 P2/P3 设计。仅使用 Depthwise Conv 提取局部纹理，
# 使用 ECA (1D Conv) 进行通道重标定。
# 绝对禁止使用 Spatial Attention (如大核池化)，以完美保留 RCSFusion 所需的物理空间对比度。
# ==============================================================================

class ECA(nn.Module):
    """Efficient Channel Attention: 纯通道重标定，不破坏空间方差"""

    def __init__(self, c, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # [b, c, h, w] -> [b, c, 1, 1] -> [b, 1, c]
        y = self.avg_pool(x).squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        # [b, 1, c] -> [b, c, 1, 1]
        y = y.transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y)


class HF_DetailExtractor(nn.Module):
    """浅层高频保真提取器 (应用于 P2, P3)"""

    def __init__(self, c1, c2):
        super().__init__()
        # 如果输入输出通道不一致，进行 1x1 线性投影
        self.proj = nn.Conv2d(c1, c2, 1, 1, 0, bias=False) if c1 != c2 else nn.Identity()

        # 3x3 深度可分离卷积：极低参数量，专注于单通道的局部像素突变(如麻点、划痕边缘)
        self.dwconv = nn.Conv2d(c2, c2, 3, 1, 1, groups=c2, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

        # 纯通道注意力，过滤无用的背景通道，保留缺陷通道
        self.eca = ECA(c2)

    def forward(self, x):
        x = self.proj(x)
        # 提取局部高频特征
        out = self.act(self.bn(self.dwconv(x)))
        # 科学严谨性：使用残差直连 (x + out)，强制保留原始图像未被修改的基础方差底噪
        return x + self.eca(out)


# ==============================================================================
# 2. 中层宏观上下文增强器 (Dilated Context Enhancer)
# 科学目的：针对中层 P4 设计。利用下采样后的特征图尺寸较小的特性，
# 使用并联的多尺度空洞卷积 (d=1, 2, 3) 捕捉大型模糊缺陷(如压痕、水渍)的宏观轮廓。
# ==============================================================================

class SEBlock(nn.Module):
    """Squeeze-and-Excitation: 用于中/深层全局语义通道的筛选"""

    def __init__(self, c1, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(c1, c1 // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c1 // ratio, c1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.fc(self.avg_pool(x).view(b, c)).view(b, c, 1, 1)
        return x * y


class DilatedContextEnhancer(nn.Module):
    """多尺度空洞上下文增强器 (应用于 P4)"""

    def __init__(self, c1, c2):
        super().__init__()
        # 降维瓶颈层，严格控制参数量和 MAC (内存访问代价)
        c_hidden = c2 // 2
        self.reduce = nn.Sequential(
            nn.Conv2d(c1, c_hidden, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c_hidden),
            nn.SiLU()
        )

        # 并联多尺度空洞卷积 (使用 groups=c_hidden 变为 Depthwise 空洞，防止显存爆炸)
        self.b1 = nn.Conv2d(c_hidden, c_hidden, 3, 1, padding=1, dilation=1, groups=c_hidden, bias=False)
        self.b2 = nn.Conv2d(c_hidden, c_hidden, 3, 1, padding=2, dilation=2, groups=c_hidden, bias=False)
        self.b3 = nn.Conv2d(c_hidden, c_hidden, 3, 1, padding=3, dilation=3, groups=c_hidden, bias=False)

        # 特征聚合升维
        self.merge = nn.Sequential(
            nn.Conv2d(c_hidden * 3, c2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU()
        )

        # 语义通道重标定
        self.se = SEBlock(c2)

    def forward(self, x):
        x_red = self.reduce(x)
        # 获取三种不同感受野的上下文轮廓
        out1 = self.b1(x_red)
        out2 = self.b2(x_red)
        out3 = self.b3(x_red)

        # 拼接 -> 聚合 -> 通道筛选
        out = torch.cat([out1, out2, out3], dim=1)
        out = self.merge(out)
        return self.se(out)
###########
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    """Conv + BN + SiLU."""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        if p is None:
            p = k // 2 if isinstance(k, int) else (k[0] // 2, k[1] // 2)
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class GRCSFusion(nn.Module):
    """
    Generalized Recalibrated Cross-scale Semantic Fusion.

    It is designed to replace Concat in YOLO Neck.

    Input:
        x = [source_feature, target_feature]

    Output:
        torch.cat([source_feature_resized, target_feature_enhanced], dim=1)

    Meaning:
        source_feature provides compensation cues.
        target_feature is enhanced by directional compensation.

    Example:
        [[-1, 6], 1, GRCSFusion, [7, 16, 0.01]]
        means:
            source = -1
            target = 6
            k = 7
            reduction = 16
            init_alpha = 0.01
    """
    def __init__(self, c_source, c_target, k=7, reduction=16, init_alpha=0.01):
        super().__init__()

        self.c_source = c_source
        self.c_target = c_target

        # Align source feature to target channels for compensation.
        self.source_align = ConvBNAct(c_source, c_target, k=1, s=1)

        # Light target projection, keeping target representation stable.
        self.target_proj = ConvBNAct(c_target, c_target, k=1, s=1)

        # Source-derived compensation branch.
        self.comp_branch = nn.Sequential(
            ConvBNAct(c_target, c_target, k=k, s=1, g=c_target),
            ConvBNAct(c_target, c_target, k=1, s=1)
        )

        # Cross-scale interaction gate.
        self.cross_gate = nn.Sequential(
            ConvBNAct(c_target * 2, c_target, k=1, s=1),
            nn.Conv2d(c_target, c_target, kernel_size=3, stride=1, padding=1, groups=c_target, bias=False),
            nn.BatchNorm2d(c_target),
            nn.Sigmoid()
        )

        # Channel calibration.
        hidden = max(c_target // reduction, 8)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c_target, hidden, kernel_size=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, c_target, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        # Spatial calibration.
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3, bias=False),
            nn.Sigmoid()
        )

        # Small residual coefficient. Using tanh keeps the scale bounded.
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))

    def forward(self, x):
        assert isinstance(x, (list, tuple)) and len(x) == 2, \
            "GRCSFusion expects two inputs: [source_feature, target_feature]."

        source, target = x

        # Resize source to target spatial size.
        if source.shape[-2:] != target.shape[-2:]:
            source_resized = F.interpolate(source, size=target.shape[-2:], mode="nearest")
        else:
            source_resized = source

        source_aligned = self.source_align(source_resized)
        target_proj = self.target_proj(target)

        # Source provides directional compensation information.
        comp = self.comp_branch(source_aligned)

        # Cross-scale gate from source-target interaction.
        gate = self.cross_gate(torch.cat([source_aligned, target_proj], dim=1))

        # Channel gate and spatial gate.
        base = source_aligned + target_proj
        c_gate = self.channel_gate(base)

        avg_map = torch.mean(base, dim=1, keepdim=True)
        max_map, _ = torch.max(base, dim=1, keepdim=True)
        s_gate = self.spatial_gate(torch.cat([avg_map, max_map], dim=1))

        # Directional residual compensation.
        scale = torch.tanh(self.alpha)
        target_enhanced = target + scale * comp * gate * c_gate * s_gate

        # Keep output channels the same as normal Concat:
        # source channels + target channels.
        return torch.cat([source_resized, target_enhanced], dim=1)
############
import math
import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv


class MambaBridgeLite(nn.Module):
    """
    MambaBridgeLite: PyTorch-only Mamba-like spatial state bridge.

    It does not depend on mamba-ssm or causal-conv1d.
    Designed for YOLOv11n industrial defect detection.

    Function:
    - Models horizontal and vertical spatial continuity.
    - Lightweight residual design.
    - Suitable before RCSFusion or after PAN fusion.
    """

    def __init__(self, c1, c2, hidden_ratio=0.25, k=7, max_scale=0.06, init_scale=0.01):
        super().__init__()

        self.proj = Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity()

        hidden_dim = max(32, int(c2 * hidden_ratio))

        self.reduce = Conv(c2, hidden_dim, 1, 1)

        # Horizontal / vertical long-range depthwise modeling
        self.dw_h = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, k), padding=(0, k // 2),
                      groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True)
        )

        self.dw_v = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(k, 1), padding=(k // 2, 0),
                      groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True)
        )

        # State gate: decide where the long-range response is reliable
        self.gate = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.Sigmoid()
        )

        self.expand = Conv(hidden_dim, c2, 1, 1)

        # Bounded residual scale
        init_ratio = init_scale / max_scale
        init_ratio = min(max(init_ratio, 1e-4), 1 - 1e-4)
        self.alpha = nn.Parameter(
            torch.tensor(math.log(init_ratio / (1 - init_ratio)), dtype=torch.float32)
        )
        self.max_scale = max_scale

    def forward(self, x):
        x = self.proj(x)

        z = self.reduce(x)

        h = self.dw_h(z)
        v = self.dw_v(z)

        state = h + v
        g = self.gate(state)

        out = self.expand(state * g)

        scale = self.max_scale * torch.sigmoid(self.alpha)

        return x + scale * out

##############
class RCSFusion(nn.Module):
    """
    RCS-Fusion: Recall-preserving Cross-Scale Fusion calibration.

    This module is designed for the neck stage after the first P5-to-P4 fusion.

    It does not replace the original fusion structure.
    It only applies a lightweight residual calibration to recover weak defect responses
    that may be diluted during cross-scale feature aggregation.

    Formula:
        F_out = F + scale * tanh(alpha) * G(F) * D(F)

    where:
        G(F) is a lightweight recall-aware spatial gate.
        D(F) is a depthwise residual detail branch.
        alpha is initialized to 0, so the module starts as an identity mapping.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        scale: float = 0.1,
    ):
        super().__init__()

        self.proj = Conv(c1, c2, 1) if c1 != c2 else nn.Identity()

        # Detail residual branch.
        self.detail = nn.Sequential(
            Conv(c2, c2, 3, 1, g=c2),
            Conv(c2, c2, 1, 1),
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)

        avg_x = torch.mean(x, dim=1, keepdim=True)
        max_x, _ = torch.max(x, dim=1, keepdim=True)
        std_x = torch.std(x, dim=1, keepdim=True, unbiased=False)

        smooth_x = F.avg_pool2d(
            avg_x,
            kernel_size=5,
            stride=1,
            padding=2,
            count_include_pad=False,
        )




        g = self.gate(gate_in)

        d = self.detail(x)

        gamma = self.scale * torch.tanh(self.alpha)

        out = x + gamma * g * d

        return out
################################
class LCRB(nn.Module):
    """
    Local Contrast Residual Block.
    It enhances weak local contrast while preserving the original feature distribution.
    Suitable for coupling with RCSFusion.
    """

    def __init__(self, c1, c2, k=3):
        super().__init__()

        # Align channels if needed
        self.proj = Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity()

        # Local average for local contrast extraction
        self.local_avg = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

        # Lightweight local texture modeling
        self.dwconv = nn.Sequential(
            nn.Conv2d(c2, c2, k, 1, k // 2, groups=c2, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU()
        )

    def forward(self, x):
        x = self.proj(x)

        # Local contrast: highlight weak local texture differences
        contrast = x - self.local_avg(x)

        out = self.dwconv(contrast)
        out = self.pwconv(out)



        return x + torch.tanh(self.alpha) * out
