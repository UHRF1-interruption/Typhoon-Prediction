"""云图 + 时序融合的强度预测模型。

结构（双分支融合）:
    标量序列 (B, T, F) ──► 线性投影 + 位置编码 ──► Transformer Encoder ──► 池化
                                                                          │
    云图 patch (B, T, 4, S, S) ─► 逐帧 CNN ─► 帧特征 (B, T, D) ─► ────────┤
                                                                          ▼
                                                          融合 → 多层感知 → 强度增量

设计要点:
  * 残差预测：输出相对最后一帧强度的增量，初始化时输出为 0，
    等价于"强度维持不变"，避免一上来就劣于持续法基线；
  * 云图分支可选：没有云图时退化为纯时序模型，便于做消融对比；
  * 帧级 CNN 共享权重，参数量小，适合 43 个台风的小样本。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class FrameEncoder(nn.Module):
    """把单帧多通道云图编码成特征向量（逐帧共享权重）。"""

    def __init__(self, in_chans: int = 4, feat_dim: int = 64, base: int = 16,
                 dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_chans, base, 3, padding=1), nn.BatchNorm2d(base), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                   # S/2
            nn.Conv2d(base, base * 2, 3, padding=1), nn.BatchNorm2d(base * 2), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                   # S/4
            nn.Conv2d(base * 2, base * 4, 3, padding=1), nn.BatchNorm2d(base * 4), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base * 4, feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B*T, C, S, S) -> (B*T, feat_dim)"""
        return self.net(x)


class IntensityFusionTransformer(nn.Module):
    """标量时序 + 云图帧特征 的融合强度预测模型。"""

    def __init__(
        self,
        scalar_dim: int,
        out_len: int,
        in_len: int,
        patch_chans: int = 4,
        patch_size: int = 32,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.15,
        use_patch: bool = True,
        img_feat: int = 64,
    ):
        super().__init__()
        self.out_len = out_len
        self.use_patch = use_patch
        self.in_len = in_len

        if d_model % nhead != 0:
            raise ValueError(f"d_model({d_model}) 必须能被 nhead({nhead}) 整除")

        self.scalar_proj = nn.Sequential(
            nn.Linear(scalar_dim, d_model), nn.LayerNorm(d_model),
        )
        self.pos = nn.Parameter(torch.randn(1, in_len, d_model) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers,
                                             enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)

        fused_dim = d_model
        if use_patch:
            self.frame_encoder = FrameEncoder(patch_chans, img_feat, dropout=dropout)
            self.img_proj = nn.Linear(img_feat, d_model)
            fused_dim = d_model * 2            # 标量池化 + 云图池化

        self.head = nn.Sequential(
            nn.Linear(fused_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, out_len),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # 输出层零初始化 -> 初始预测恒为 0 增量（强度维持）
        last = self.head[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, s: torch.Tensor, p: torch.Tensor | None = None,
                key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """s: (B, T, F) 标量序列；p: (B, T, C, S, S) 云图 patch。

        返回 (B, out_len) 强度增量 (m/s)。
        """
        h = self.scalar_proj(s) + self.pos[:, : s.size(1)]
        h = self.encoder(h, src_key_padding_mask=key_padding_mask)
        h = self.norm(h)
        # 掩码均值池化（避免 padding 影响）
        if key_padding_mask is None:
            pooled = h.mean(dim=1)
        else:
            valid = (~key_padding_mask).unsqueeze(-1).to(h.dtype)
            pooled = (h * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)

        if self.use_patch:
            if p is None:
                raise ValueError("模型启用了云图分支，但未提供 patch")
            b, t, c, hs, ws = p.shape
            frames = self.frame_encoder(p.reshape(b * t, c, hs, ws))
            frames = frames.view(b, t, -1)
            img = self.img_proj(frames).mean(dim=1)
            pooled = torch.cat([pooled, img], dim=-1)

        return self.head(pooled)


class IntensityRnnBaseline(nn.Module):
    """纯时序 GRU 基线（不看云图），用于验证云图是否带来增益。"""

    def __init__(self, scalar_dim: int, out_len: int, hidden: int = 128,
                 num_layers: int = 2, dropout: float = 0.1, **_: object):
        super().__init__()
        self.out_len = out_len
        self.use_patch = False
        self.gru = nn.GRU(scalar_dim, hidden, num_layers=num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, out_len),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, s: torch.Tensor, p: torch.Tensor | None = None,
                key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        if key_padding_mask is not None:
            s = s.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        _, h = self.gru(s)
        return self.head(h[-1])


class PersistenceIntensity(nn.Module):
    """强度持续法：假设强度不变（增量恒为 0）。"""

    def __init__(self, scalar_dim: int, out_len: int, **_: object):
        super().__init__()
        self.out_len = out_len
        self.use_patch = False

    def forward(self, s: torch.Tensor, p: torch.Tensor | None = None,
                key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        return torch.zeros(s.size(0), self.out_len, device=s.device, dtype=s.dtype)


class IntensityFeatureTransformer(nn.Module):
    """云图手工特征 + 标量序列的融合模型（低参数版）。

    与 IntensityFusionTransformer 的区别:
      * 云图不再过 CNN，而是每帧 15 维物理特征（眼区对比度、冷云面积等），
        参数量从约 30 万降到几乎为 0，显著缓解小样本过拟合；
      * 云图特征与标量特征在同一 Transformer 内做跨模态注意力。

    输入:
      s: (B, T, F_s) 标量序列
      img: (B, T, F_img) 云图特征序列
    """

    def __init__(self, scalar_dim: int, out_len: int, in_len: int,
                 img_dim: int = 15, d_model: int = 128, nhead: int = 8,
                 num_layers: int = 2, dim_feedforward: int = 256,
                 dropout: float = 0.25, use_patch: bool = True, **_: object):
        super().__init__()
        self.out_len = out_len
        self.use_patch = use_patch
        self.use_image_features = True
        if d_model % nhead != 0:
            raise ValueError(f"d_model({d_model}) 必须能被 nhead({nhead}) 整除")

        fused_in = scalar_dim + (img_dim if use_patch else 0)
        self.input_proj = nn.Sequential(
            nn.Linear(fused_in, d_model), nn.LayerNorm(d_model),
        )
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
            nn.Linear(d_model, out_len),
        )
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, s: torch.Tensor, p: torch.Tensor | None = None,
                key_padding_mask: torch.Tensor | None = None,
                img_feat: torch.Tensor | None = None) -> torch.Tensor:
        """p 为云图 patch（本模型不用）；img_feat 为预计算的特征序列。"""
        x = s
        if self.use_patch:
            if img_feat is None:
                raise ValueError("特征版模型需要 img_feat（云图手工特征序列）")
            x = torch.cat([s, img_feat], dim=-1)
        h = self.input_proj(x) + self.pos[:, : x.size(1)]
        h = self.encoder(h, src_key_padding_mask=key_padding_mask)
        h = self.norm(h)
        if key_padding_mask is None:
            pooled = h.mean(dim=1)
        else:
            valid = (~key_padding_mask).unsqueeze(-1).to(h.dtype)
            pooled = (h * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        return self.head(pooled)


def build_intensity_model(cfg, scalar_dim: int, in_len: int, out_len: int,
                          use_patch: bool, patch_size: int,
                          patch_chans: int = 4, img_dim: int | None = None) -> nn.Module:
    """按配置构建强度预测模型。

    img_dim: 云图手工特征维数。为 None 时退回配置值/默认 15；
             训练脚本会从实际 pfeat 形状传入，避免特征表扩充后维度错配。
    """
    name = str(getattr(cfg, "name", "fusion_transformer")).lower()
    if name in ("feature_transformer", "feat_transformer"):
        return IntensityFeatureTransformer(
            scalar_dim=scalar_dim, out_len=out_len, in_len=in_len,
            img_dim=int(img_dim if img_dim is not None
                        else getattr(cfg, "img_feat_dim", 15)),
            d_model=int(getattr(cfg, "d_model", 128)),
            nhead=int(getattr(cfg, "nhead", 8)),
            num_layers=int(getattr(cfg, "num_layers", 2)),
            dim_feedforward=int(getattr(cfg, "dim_feedforward", 256)),
            dropout=float(getattr(cfg, "dropout", 0.25)),
            use_patch=use_patch,
        )
    if name in ("fusion_transformer", "transformer"):
        return IntensityFusionTransformer(
            scalar_dim=scalar_dim, out_len=out_len, in_len=in_len,
            patch_chans=patch_chans, patch_size=patch_size,
            d_model=int(getattr(cfg, "d_model", 128)),
            nhead=int(getattr(cfg, "nhead", 8)),
            num_layers=int(getattr(cfg, "num_layers", 3)),
            dim_feedforward=int(getattr(cfg, "dim_feedforward", 256)),
            dropout=float(getattr(cfg, "dropout", 0.15)),
            use_patch=use_patch,
            img_feat=int(getattr(cfg, "img_feat", 64)),
        )
    if name in ("dual_head", "dual"):
        return IntensityDualHeadTransformer(
            scalar_dim=scalar_dim, out_len=out_len, in_len=in_len,
            mw_chans=patch_chans, mw_size=patch_size,
            hidden=int(getattr(cfg, "hidden", 96)),
            num_layers=int(getattr(cfg, "num_layers", 2)),
            dropout=float(getattr(cfg, "dropout", 0.2)),
            img_feat=int(getattr(cfg, "img_feat", 64)),
        )
    if name in ("mw_transformer", "microwave"):
        return IntensityMicrowaveTransformer(
            scalar_dim=scalar_dim, out_len=out_len, in_len=in_len,
            mw_chans=patch_chans, mw_size=patch_size,
            hidden=int(getattr(cfg, "hidden", 96)),
            num_layers=int(getattr(cfg, "num_layers", 2)),
            dropout=float(getattr(cfg, "dropout", 0.2)),
            img_feat=int(getattr(cfg, "img_feat", 64)),
        )
    if name == "gru":
        return IntensityRnnBaseline(scalar_dim, out_len,
                                    hidden=int(getattr(cfg, "d_model", 128)))
    if name == "persistence":
        return PersistenceIntensity(scalar_dim, out_len)
    raise ValueError(f"未知强度模型: {name}，可选 fusion_transformer/gru/persistence")


class IntensityMicrowaveTransformer(nn.Module):
    """GRU 时序 + 单帧微波 patch 的融合模型。

    与 IntensityFusionTransformer 的区别：
      * 微波是**单帧锚观测**（不是 8 帧序列），因为微波是离散快照
        （每台风中位 44 次、间隔中位 2.4 h），强行拼成序列会丢掉绝大部分样本；
      * 额外输入「锚观测与窗口末端的时间差 dt」和「通道可信度 mask」，
        让模型知道这条观测有多新、哪些通道是替代值。

    输入:
        s        (B, T, F_s)     标量序列
        mw       (B, C, S, S)    锚观测 patch（K）
        mw_dt    (B,)            时间差（小时，0-3）
        mw_mask  (B, C)          0=原始, 1=缺失/已替代
    输出:
        (B, out_len)             强度残差

    **使用约束**：本模型假定**每个样本都有微波**。若样本里存在"没有微波"
    的（全零 patch），零图会污染 CNN 的 BatchNorm running stats，
    让推理时真实 patch 的输出被系统性拉偏。混合样本请用
    IntensityDualHeadTransformer（它把无微波样本路由到标量头，
    CNN 只接触真实 patch）。
    """

    def __init__(self, scalar_dim: int, out_len: int, in_len: int,
                 mw_chans: int = 4, mw_size: int = 64,
                 hidden: int = 96, num_layers: int = 2, dropout: float = 0.2,
                 img_feat: int = 64, **_: object):
        super().__init__()
        self.out_len = out_len
        self.use_patch = True          # 表示"用了图像分支"，供外部判断
        self.gru = nn.GRU(scalar_dim, hidden, num_layers=num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.frame_encoder = FrameEncoder(mw_chans, img_feat, dropout=dropout)
        self.img_proj = nn.Linear(img_feat, hidden)
        # dt 与 mask 一起进一个小 MLP，再与图像特征相加
        self.aux_proj = nn.Sequential(
            nn.Linear(1 + mw_chans, hidden), nn.GELU(), nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden * 2), nn.Linear(hidden * 2, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, out_len),
        )

    def forward(self, s, p=None, mw=None, mw_dt=None, mw_mask=None,
                key_padding_mask=None):
        if key_padding_mask is not None:
            s = s.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        _, h = self.gru(s)
        seq = h[-1]
        if mw is None:
            # 没有微波的样本：图像分支给零，靠 aux 的 mask 全 1 表达"不可用"
            b = s.size(0)
            img = torch.zeros(b, self.img_proj.out_features, device=s.device)
            aux = torch.zeros(b, self.aux_proj[0].out_features, device=s.device)
        else:
            # 标准化由 Dataset 完成（逐通道 z-score）
            img = self.img_proj(self.frame_encoder(mw))
            dt = torch.nan_to_num(mw_dt, nan=0.0).unsqueeze(-1).to(s.dtype)
            aux = self.aux_proj(torch.cat([dt, mw_mask.to(s.dtype)], dim=-1))
        return self.head(torch.cat([seq, img + aux], dim=-1))

class IntensityDualHeadTransformer(nn.Module):
    """共享 GRU 编码器 + 双头：有微波走融合头，无微波走纯标量头。

    为什么这样设计
    --------------
    微波是**离散观测**：全量 15,126 个样本里只有 34.4% 恰好带有微波
    （每台风中位 44 次，间隔中位 2.4 h）。之前的做法是把缺观测的样本
    的 patch 填零，结果 65.6% 的样本一直在喂一个"全零标记"，
    污染了共享表征——但那次对照只跑了 3 epoch，结论不可用（见报告 §7.4）。

    本模型用**路由**替代填充：
        has_mw=True  -> fusion_head（GRU 输出 + 微波特征 + 时间差）
        has_mw=False -> scalar_head（只用 GRU 输出）

    **标量头与 IntensityRnnBaseline 完全同构**。因此在"全部样本都没有微波"
    的极限下，本模型**逐位退化为 GRU 基线**——对照是干净的，任何差异
    都只能来自微波样本的那条分支。

    梯度分配
    --------
    `torch.where` 使融合头只从带微波样本获得梯度，份额约 1/3。
    训练时应配合损失加权（见 smoke_microwave.py 的 --mw-loss-weight），
    否则融合头欠拟合会被误读成"架构更差"——这正是 §7.4 记录的陷阱。
    """

    def __init__(self, scalar_dim: int, out_len: int, in_len: int,
                 mw_chans: int = 4, mw_size: int = 64,
                 hidden: int = 96, num_layers: int = 2, dropout: float = 0.2,
                 img_feat: int = 64, **_: object):
        super().__init__()
        self.out_len = out_len
        self.use_patch = True
        self.gru = nn.GRU(scalar_dim, hidden, num_layers=num_layers,
                          batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)

        # 标量头：与 IntensityRnnBaseline 逐层同构（保证可退化性）
        self.scalar_head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, out_len),
        )
        nn.init.zeros_(self.scalar_head[-1].weight)
        nn.init.zeros_(self.scalar_head[-1].bias)

        # 融合头
        self.frame_encoder = FrameEncoder(mw_chans, img_feat, dropout=dropout)
        self.img_proj = nn.Linear(img_feat, hidden)
        self.aux_proj = nn.Sequential(
            nn.Linear(1 + mw_chans, hidden), nn.GELU(), nn.Dropout(dropout),
        )
        self.fusion_head = nn.Sequential(
            nn.LayerNorm(hidden * 2), nn.Linear(hidden * 2, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, out_len),
        )
        nn.init.zeros_(self.fusion_head[-1].weight)
        nn.init.zeros_(self.fusion_head[-1].bias)

    def forward(self, s, p=None, mw=None, mw_dt=None, mw_mask=None,
                has_mw=None, key_padding_mask=None):
        if key_padding_mask is not None:
            s = s.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        _, h = self.gru(s)
        seq = h[-1]

        out_scalar = self.scalar_head(seq)
        if mw is None:
            return out_scalar
        if has_mw is None:
            has_mw = torch.ones(mw.size(0), dtype=torch.bool, device=mw.device)

        # 只对**真有微波**的样本跑 CNN 与融合头。
        # 绝不能图省事把零填充 patch 一起喂进去：BatchNorm 会把 65.6% 的
        # 全零假图算进 running stats（实测 running_mean 的漂移方向与量级
        # 都与纯真实 batch 不同），推理时真实 patch 的输出被系统性拉偏。
        # 这会让"双头更差"看起来像架构问题，实际是归一化统计被污染。
        idx = has_mw.nonzero(as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            return out_scalar
        mw_sub = mw.index_select(0, idx)
        dt_sub = torch.nan_to_num(mw_dt.index_select(0, idx), nan=0.0)
        mask_sub = mw_mask.index_select(0, idx)

        img = self.img_proj(self.frame_encoder(mw_sub))
        aux = self.aux_proj(torch.cat([dt_sub.unsqueeze(-1).to(s.dtype),
                                       mask_sub.to(s.dtype)], dim=-1))
        fused = self.fusion_head(
            torch.cat([seq.index_select(0, idx), img + aux], dim=-1))
        # index_put 是 out-of-place 的，梯度分别回到标量头与融合头
        return out_scalar.index_put((idx,), fused)
