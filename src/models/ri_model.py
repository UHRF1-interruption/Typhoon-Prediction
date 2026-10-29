"""RI（快速增强）事件分类模型。

三个对照:
  RIFusionNet     : 标量序列 + 云图（CNN 或手工特征二选一）→ 二分类
  RISequenceGRU   : 纯时序 GRU（只看标量与历史强度）
  RIPersistence   : 持续法基线 —— 用「当前强度变化率」外推，
                    即把历史最后两步的强度差当作未来 24h 的预期增幅

为什么要持续法基线: RI 判定依赖强度变化趋势。如果模型连「按当前趋势线性外推」
都赢不了，说明它没有学到任何关于强度突变的额外信息。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .intensity_transformer import FrameEncoder


class RIFusionNet(nn.Module):
    """标量序列 + 云图特征 → RI 二分类 logit。

    云图输入二选一:
      use_cnn=True  -> 吃 patch (B,T,C,S,S)，过共享 CNN
      use_cnn=False -> 吃手工特征 (B,T,F_img)
    这样可以在同一套代码里做「CNN vs 物理特征」的公平对比。
    """

    def __init__(
        self,
        scalar_dim: int,
        img_dim: int | None = None,
        d_model: int = 96,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 192,
        dropout: float = 0.3,
        use_cnn: bool = False,
        patch_chans: int = 3,
        cnn_feat: int = 48,
        in_len: int = 8,
    ):
        super().__init__()
        self.use_cnn = use_cnn
        self.use_image = use_cnn or (img_dim is not None and img_dim > 0)
        if d_model % nhead != 0:
            raise ValueError(f"d_model({d_model}) 必须能被 nhead({nhead}) 整除")

        if use_cnn:
            self.frame_encoder = FrameEncoder(patch_chans, cnn_feat, dropout=dropout)
            img_in = cnn_feat
        else:
            img_in = img_dim or 0

        fused_in = scalar_dim + (img_in if self.use_image else 0)
        self.input_proj = nn.Sequential(nn.Linear(fused_in, d_model), nn.LayerNorm(d_model))
        self.pos = nn.Parameter(torch.randn(1, in_len, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers,
                                             enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, 1),          # 输出 logit
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, s: torch.Tensor, p: torch.Tensor | None = None,
                pfeat: torch.Tensor | None = None) -> torch.Tensor:
        x = s
        if self.use_image:
            if self.use_cnn:
                if p is None:
                    raise ValueError("use_cnn=True 需要 patch 输入 p")
                b, t, c, hs, ws = p.shape
                img = self.frame_encoder(p.reshape(b * t, c, hs, ws)).view(b, t, -1)
            else:
                if pfeat is None:
                    raise ValueError("特征模式需要 pfeat 输入")
                img = pfeat
            x = torch.cat([s, img], dim=-1)
        h = self.input_proj(x) + self.pos[:, : x.size(1)]
        h = self.encoder(h)
        h = self.norm(h)
        return self.head(h.mean(dim=1)).squeeze(-1)     # (B,)


class RISequenceGRU(nn.Module):
    """纯时序 GRU 分类器（无云图）。"""

    def __init__(self, scalar_dim: int, hidden: int = 96, num_layers: int = 2,
                 dropout: float = 0.2, **_: object):
        super().__init__()
        self.use_cnn = False
        self.use_image = False
        self.gru = nn.GRU(scalar_dim, hidden, num_layers=num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, 1),
        )

    def forward(self, s: torch.Tensor, p: torch.Tensor | None = None,
                pfeat: torch.Tensor | None = None) -> torch.Tensor:
        _, h = self.gru(s)
        return self.head(h[-1]).squeeze(-1)


class RIPersistence(nn.Module):
    """持续法基线：按输入窗口末段的强度变化率线性外推 24h。

    这不是神经网络，输出的是一个可学习外的"规则分数"，用于校准下限。
    实际打分由 train_ri.py 直接计算（见 persistence_score）。
    """

    def __init__(self, **_: object):
        super().__init__()
        self.use_cnn = False
        self.use_image = False

    def forward(self, s: torch.Tensor, p: torch.Tensor | None = None,
                pfeat: torch.Tensor | None = None) -> torch.Tensor:
        # s[..., 0] 是 wind，最后两差的差就是当前变化率（每 3h）
        rate = s[:, -1, 0] - s[:, -2, 0] if s.size(1) > 1 else torch.zeros(s.size(0), device=s.device)
        return rate * 8.0          # 外推 8 步（24h）


def build_ri_model(cfg, scalar_dim: int, img_dim: int, in_len: int,
                   patch_chans: int = 3, name: str | None = None) -> nn.Module:
    """构建 RI 模型。

    name 显式指定模型种类（优先于 cfg.name），这样同一个超参配置可以
    复用到多个模型上，避免"配置里的 name 掩盖了真实要建的模型"这类 bug。
    """
    name = str(name if name is not None else cfg.name).lower()
    if name in ("fusion_transformer", "ri_fusion"):
        return RIFusionNet(
            scalar_dim=scalar_dim, img_dim=img_dim, in_len=in_len,
            d_model=int(getattr(cfg, "d_model", 96)),
            nhead=int(getattr(cfg, "nhead", 4)),
            num_layers=int(getattr(cfg, "num_layers", 2)),
            dim_feedforward=int(getattr(cfg, "dim_feedforward", 192)),
            dropout=float(getattr(cfg, "dropout", 0.3)),
            use_cnn=bool(getattr(cfg, "use_cnn", False)),
            patch_chans=patch_chans,
        )
    if name in ("feature_transformer", "feat"):
        return RIFusionNet(
            scalar_dim=scalar_dim, img_dim=img_dim, in_len=in_len,
            d_model=int(getattr(cfg, "d_model", 96)),
            nhead=int(getattr(cfg, "nhead", 4)),
            num_layers=int(getattr(cfg, "num_layers", 2)),
            dim_feedforward=int(getattr(cfg, "dim_feedforward", 192)),
            dropout=float(getattr(cfg, "dropout", 0.3)),
            use_cnn=False, patch_chans=patch_chans,
        )
    if name in ("cnn_transformer", "cnn"):
        return RIFusionNet(
            scalar_dim=scalar_dim, img_dim=img_dim, in_len=in_len,
            d_model=int(getattr(cfg, "d_model", 96)),
            nhead=int(getattr(cfg, "nhead", 4)),
            num_layers=int(getattr(cfg, "num_layers", 2)),
            dim_feedforward=int(getattr(cfg, "dim_feedforward", 192)),
            dropout=float(getattr(cfg, "dropout", 0.3)),
            use_cnn=True, patch_chans=patch_chans,
        )
    if name == "gru":
        return RISequenceGRU(scalar_dim=scalar_dim, hidden=int(getattr(cfg, "d_model", 96)),
                             dropout=float(getattr(cfg, "dropout", 0.2)))
    if name == "persistence":
        return RIPersistence()
    raise ValueError(f"未知 RI 模型: {name}")
