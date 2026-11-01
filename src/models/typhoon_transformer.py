"""台风路径预测模型集合。

- TyphoonTransformer : Encoder + 可学习 query 的轻量 Decoder (DETR 式)
- GRUBaseline        : 编码器换成 GRU 的对照模型
- PersistenceBaseline: 持续法（原地不动），气象业务最常用的对照基线
- LinearBaseline     : 线性回归，检查深度学习是否真的带来增益

所有模型输入  (B, in_len, F)
所有模型输出  (B, out_len, 2) —— 表示相对最后观测点的增量 [dlat, dlon]
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """标准正弦位置编码。"""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.0):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


class TyphoonTransformer(nn.Module):
    """Encoder-only 主干 + 可学习未来 query 解码当前 sequence length。

    设计要点:
    1. pred_mode='delta' 时只预测位移增量，避免直接回归绝对经纬度带来的尺度问题；
    2. 使用 key_padding_mask 屏蔽轨迹开头的 padding；
    3. 输出头初始化为 0，让模型从"持续法"基线开始学，训练更稳。
    """

    def __init__(
        self,
        feat_dim: int,
        out_len: int,
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        pred_mode: str = "delta",
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model({d_model}) 必须能被 nhead({nhead}) 整除")

        self.out_len = out_len
        self.pred_mode = pred_mode

        self.input_proj = nn.Sequential(
            nn.Linear(feat_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,      # Pre-LN，训练更稳定
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers,
                                             enable_nested_tensor=False)

        # 未来时刻的可学习 query（类似 DETR 的 object queries）
        self.query_embed = nn.Parameter(torch.randn(1, out_len, d_model) * 0.02)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 2)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # 输出头零初始化 -> 初期预测恒为 0 增量 = 持续法基线
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None,
                last_lat: torch.Tensor | None = None,
                last_lon: torch.Tensor | None = None) -> torch.Tensor:
        """x: (B, in_len, F) -> (B, out_len, 2)

        key_padding_mask: (B, in_len) bool，True 表示该位置是 padding。
        last_lat/last_lon: 最后输入点的绝对坐标。纯学习模型用不到，
            但物理约束版（TyphoonTransformerPhysics）需要它把 km 残差界
            换算成经纬度增量。统一接口，避免训练器分支。
        """
        h = self.input_proj(x)
        h = self.pos_enc(h)
        memory = self.encoder(h, src_key_padding_mask=key_padding_mask)

        queries = self.query_embed.expand(x.size(0), -1, -1)
        out = self.decoder(queries, memory, memory_key_padding_mask=key_padding_mask)
        out = self.norm(out)
        return self.head(out)


class GRUBaseline(nn.Module):
    """GRU 对照模型：同样的输入输出契约，用于说明 Transformer 是否真的更好。"""

    def __init__(self, feat_dim: int, out_len: int, hidden: int = 128,
                 num_layers: int = 2, dropout: float = 0.1, pred_mode: str = "delta"):
        super().__init__()
        self.out_len = out_len
        self.pred_mode = pred_mode
        self.gru = nn.GRU(feat_dim, hidden, num_layers=num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_len * 2),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None,
                last_lat: torch.Tensor | None = None,
                last_lon: torch.Tensor | None = None) -> torch.Tensor:
        if key_padding_mask is not None:
            x = x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        _, h = self.gru(x)
        out = self.head(h[-1])
        return out.view(x.size(0), self.out_len, 2)


class PersistenceBaseline(nn.Module):
    """持续法：假设台风不动。输出恒为 0 增量。"""

    def __init__(self, feat_dim: int, out_len: int, **_: object):
        super().__init__()
        self.out_len = out_len
        self.pred_mode = "delta"

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None,
                last_lat: torch.Tensor | None = None,
                last_lon: torch.Tensor | None = None) -> torch.Tensor:
        return torch.zeros(x.size(0), self.out_len, 2, device=x.device, dtype=x.dtype)


class LinearBaseline(nn.Module):
    """线性回归：展平整个输入窗口后线性映射。"""

    def __init__(self, feat_dim: int, out_len: int, in_len: int, **_: object):
        super().__init__()
        self.out_len = out_len
        self.pred_mode = "delta"
        self.linear = nn.Linear(in_len * feat_dim, out_len * 2)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None,
                last_lat: torch.Tensor | None = None,
                last_lon: torch.Tensor | None = None) -> torch.Tensor:
        return self.linear(x.flatten(1)).view(x.size(0), self.out_len, 2)



class TyphoonTransformerPhysics(nn.Module):
    """硬约束版：预测 = **恒速物理先验 + 模长有界的学习残差**。

    形式对应 WHC-PINN 的硬约束式 `û = g(x) + ℓ(x)·N(x;θ)`：
        g(x) = 恒速先验（用最近 3 步加权速度外推）
        ℓ·N  = 有界残差，|残差| ≤ cap_km(t)，方向由网络决定
    与 WHC-PINN 的差别在于 **ℓ 取常数（= cap）而不随时效衰减**。
    这是刻意的：若 ℓ 随预报时效衰减，长时效的预测就退化为先验、
    网络在那里拿不到梯度——正是该文警告的"ℓ 归零处网络失效"，
    也是本文最该避免的**过度外推**。实测 24h 残差 p99 达 996 km，
    先验单独误差 258 km，长时效恰是最需要学习的地方。

    实测依据（20094 个滑窗）：恒速先验 4 时效平均 145.6 km，
    优于持续法 338.8 km、劣于学习模型 119.0 km —— 适合做底座。

    ⚠️ 残差界由数据 p99 定（199/434/703/996 km）。界过紧会让 tanh 饱和、
    梯度趋零，用 physics.saturation_rate 监控；超过约 5% 就说明界太紧。
    """

    def __init__(self, feat_dim: int, out_len: int, d_model: int = 128,
                 nhead: int = 8, num_encoder_layers: int = 4,
                 num_decoder_layers: int = 2, dim_feedforward: int = 512,
                 dropout: float = 0.1, pred_mode: str = "delta",
                 cap_km: tuple[float, ...] | None = None,
                 prior_weights: tuple[float, ...] = (0.2, 0.3, 0.5)):
        super().__init__()
        from ..training.physics import RESIDUAL_CAP_KM, bounded_residual, constant_velocity_prior

        self._bounded_residual = staticmethod(bounded_residual)
        self._prior = staticmethod(constant_velocity_prior)
        self.out_len = out_len
        self.pred_mode = pred_mode
        self.prior_weights = prior_weights
        cap = cap_km or RESIDUAL_CAP_KM
        if len(cap) != out_len:
            raise ValueError(f"cap_km 长度({len(cap)}) 必须等于 out_len({out_len})")
        self.register_buffer("cap_km", torch.tensor(cap, dtype=torch.float32))

        self.backbone = TyphoonTransformer(
            feat_dim=feat_dim, out_len=out_len, d_model=d_model, nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward, dropout=dropout, pred_mode=pred_mode,
        )
        # 残差头零初始化 -> 训练初期严格等于恒速先验
        # （对照：纯学习模型零初始化 = 持续法。这样起点就是更好的物理基线）
        nn.init.zeros_(self.backbone.head.weight)
        nn.init.zeros_(self.backbone.head.bias)
        self.last_raw: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None,
                last_lat: torch.Tensor | None = None,
                last_lon: torch.Tensor | None = None) -> torch.Tensor:
        if last_lat is None:
            raise ValueError("物理约束模型需要 last_lat（用于 km<->度换算）")
        prior = self._prior(x, self.out_len, self.prior_weights)
        raw = self.backbone(x, key_padding_mask=key_padding_mask)
        self.last_raw = raw.detach()
        return prior + self._bounded_residual(raw, last_lat, self.cap_km)

def build_model(cfg, feat_dim: int, in_len: int, out_len: int) -> nn.Module:
    """按配置构建模型。"""
    name = str(cfg.model.name).lower()
    common = dict(feat_dim=feat_dim, out_len=out_len)
    if name == "transformer":
        return TyphoonTransformer(
            **common,
            d_model=cfg.model.d_model,
            nhead=cfg.model.nhead,
            num_encoder_layers=cfg.model.num_encoder_layers,
            num_decoder_layers=cfg.model.num_decoder_layers,
            dim_feedforward=cfg.model.dim_feedforward,
            dropout=cfg.model.dropout,
            pred_mode=cfg.model.pred_mode,
        )
    if name in ("transformer_physics", "physics"):
        return TyphoonTransformerPhysics(
            **common,
            d_model=cfg.model.d_model,
            nhead=cfg.model.nhead,
            num_encoder_layers=cfg.model.num_encoder_layers,
            num_decoder_layers=cfg.model.num_decoder_layers,
            dim_feedforward=cfg.model.dim_feedforward,
            dropout=cfg.model.dropout,
            pred_mode=cfg.model.pred_mode,
            cap_km=getattr(cfg.model, "cap_km", None),
        )
    if name == "gru":
        return GRUBaseline(**common, hidden=cfg.model.d_model,
                           dropout=cfg.model.dropout, pred_mode=cfg.model.pred_mode)
    if name == "persistence":
        return PersistenceBaseline(**common)
    if name == "linear":
        return LinearBaseline(**common, in_len=in_len)
    raise ValueError(f"未知模型: {name}，可选 transformer/transformer_physics/gru/persistence/linear")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
