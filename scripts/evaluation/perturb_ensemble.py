"""扰动集合的评估：受约束扰动能否给出可信的不确定性。

做什么
------
用训练好的路径模型 + `src/training/perturb.py` 的受约束扰动生成 K 个集合成员，
评估三件事：

1. **集合均值 vs 控制预报**：受约束扰动是对称的，均值理论上≈控制预报。
   若真的如此，说明这套扰动不提供"偏差订正"，价值只在离散度。
2. **spread-skill 关系**：集合离散度能否指示实际误差。按离散度分箱，
   若各箱的 RMSE 随离散度单调上升，说明离散度有信息量。
3. **覆盖率**：观测落在"均值 ± α·离散度"内的比例是否接近名义比例。

为什么这样评
------------
§4.11 已证明"用物理先验做输出锚点"失败（+5.2 km）。扰动集合是**另一条**用物理
的路径：不问"预测多少"，而问"不确定多大"。因此主指标不是 MAE，而是
spread-skill 与覆盖率——用 MAE 评判它会重复 §4.11 的错误。

用法
----
    python scripts/evaluation/perturb_ensemble.py \
        --ckpt outputs/phys_seed_base_0/checkpoints/best.pt \
        --eps-km 100 --members 4 --steps 10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.config import load_config  # noqa: E402
from src.data.dataset import load_bundle, split_by_year  # noqa: E402
from src.data.features import haversine_km  # noqa: E402
from src.models.typhoon_transformer import build_model  # noqa: E402
from src.training.perturb import (amp_km, grow, members, orthogonality_error,
                                  random_bounded)  # noqa: E402


def _data_path(cfg, value):
    from src.config import resolve_path
    return resolve_path(cfg, value, ROOT)


def main() -> int:
    ap = argparse.ArgumentParser(description="扰动集合评估")
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--eps-km", type=float, default=100.0,
                    help="扰动幅度（位置公里）。参考：6h 位移 p50≈115 km")
    ap.add_argument("--members", type=int, default=4, help="成员数 K（≤ 特征维度）")
    ap.add_argument("--steps", type=int, default=10, help="增长优化步数；0=纯随机正交")
    ap.add_argument("--limit", type=int, default=200, help="用多少测试样本")
    ap.add_argument("--center", choices=["control", "mean"], default="control",
                    help="集合中心。control=保留确定性子报、只用扰动刻画离散度"
                         "（默认，且是 Li et al. 2026 的用法）；"
                         "mean=成员平均，实测会显著恶化精度")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    bundle, spec = load_bundle(_data_path(cfg, cfg.data.processed_dir))
    splits = split_by_year(bundle, list(cfg.data.val_years), list(cfg.data.test_years))
    test = splits["test"]
    n = min(args.limit, len(test))
    idx = test[-n:]                                    # 测试年尾部，与训练年份分开
    # 必须用与训练相同的标准化，否则预测完全失真
    # （漏掉这一步时 24h 误差 849 km，而模型实际是 208 km）
    x_raw = (bundle["x"][idx] - spec.mean) / spec.std
    x = torch.tensor(x_raw, dtype=torch.float32)
    y_abs = bundle["y_abs"][idx]
    last = torch.tensor(bundle["last_abs"][idx], dtype=torch.float32)

    model = build_model(cfg, feat_dim=x.shape[-1], in_len=x.shape[1],
                        out_len=y_abs.shape[1])
    state = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state.get("model", state))
    model.eval()
    print(f"[载入] {args.ckpt}  {n} 个测试样本，K={args.members}，eps={args.eps_km} km")

    def predict(xm, _lat=None):
        """xm: (k,B,T,F) -> (k,B,out,2) 绝对位置。

        锚点恒为 `last`：扰动只加在**输入特征**上，不动锚点坐标。
        第二个参数仅为满足 `perturb.grow(fn, ...)` 的回调签名（它会把纬度传进来）。
        **不能加 @torch.no_grad()** —— 增长优化需要对扰动求梯度。
        """
        out = []
        for i in range(xm.shape[0]):
            d = model(xm[i], last_lat=last[:, 0], last_lon=last[:, 1])
            out.append(torch.stack([last[:, 0].unsqueeze(1) + d[..., 0],
                                    last[:, 1].unsqueeze(1) + d[..., 1]], -1))
        return torch.stack(out, 0)

    # 只有 dlat/dlon 参与扰动：物理扰动 = 标准化扰动 × std。
    # 其余特征的标准差填 1（它们不被扰动，除以 1 无副作用）。
    F = x.shape[-1]
    pos_std = torch.ones(F, dtype=torch.float32)
    pos_std[:2] = torch.tensor(spec.std[:2], dtype=torch.float32)
    lat_ref = float(last[:, 0].mean())
    if args.steps > 0:
        print(f"[扰动] 增长优化 {args.steps} 步 ...")
        deltas = grow(predict, x, last[:, 0], args.eps_km, args.members,
                      steps=args.steps, pos_std=pos_std)
        tag = "增长优化"
    else:
        # dim=2：只有 dlat/dlon 两个自由度，正交方向最多 2 个
        deltas = torch.zeros(args.members, x.shape[-1])
        sub = random_bounded(2, args.eps_km, min(args.members, 2), lat_ref)
        deltas[: sub.shape[0], :2] = sub
        if sub.shape[0] < args.members:      # 超过 2 个成员时补随机方向（不再严格正交）
            extra = random_bounded(2, args.eps_km,
                                   args.members - sub.shape[0], lat_ref, seed=99)
            deltas[sub.shape[0]:, :2] = extra
        tag = "随机正交"
    print(f"        {tag}：正交偏差 {orthogonality_error(deltas):.1e}，"
          f"位置振幅 {float(amp_km(deltas, lat_ref).max()):.1f} km")

    with torch.no_grad():
        ctrl = predict(x.unsqueeze(0))[0]                # (B,out,2)
        ens = predict(members(x, deltas, pos_std))       # (K,B,out,2)
    # 集合中心：默认用控制预报。
    # 实测（eps=100/150/250）成员平均会让 MAE 恶化 +16/+44/+137 km，
    # 因为模型对输入扰动的响应不对称，平均不能回到控制预报。
    # 扰动集合的价值在离散度，不在均值——所以默认以控制为中心。
    ens_mean = ctrl if args.center == "control" else ens.mean(0)

    def err_km(p):
        return haversine_km(p[..., 0].numpy(), p[..., 1].numpy(),
                            y_abs[..., 0], y_abs[..., 1])

    e_ctrl = err_km(ctrl)
    e_mean = err_km(ens_mean)
    # 离散度：成员到集合均值的平均球面距离（逐样本、逐时效）
    spread = np.stack([
        haversine_km(ens[k, ..., 0].numpy(), ens[k, ..., 1].numpy(),
                     ens_mean[..., 0].numpy(), ens_mean[..., 1].numpy())
        for k in range(args.members)]).mean(0)

    print("\n" + "=" * 74)
    print("1) 集合均值 vs 控制预报（逐时效 MAE, km）")
    print("=" * 74)
    leads = [6 * (i + 1) for i in range(y_abs.shape[1])]
    print(f"{'':10s}" + "".join(f"{f'{h}h':>10s}" for h in leads))
    print(f"{'控制':10s}" + "".join(f"{v:10.1f}" for v in e_ctrl.mean(0)))
    print(f"{'集合均值':10s}" + "".join(f"{v:10.1f}" for v in e_mean.mean(0)))
    print(f"{'Δ':10s}" + "".join(f"{v:+10.1f}" for v in (e_mean - e_ctrl).mean(0)))

    print("\n" + "=" * 74)
    print(f"2) spread-skill 关系（{tag}，K={args.members}）")
    print("=" * 74)
    sp = spread.ravel()
    sk = e_mean.ravel()
    qs = np.quantile(sp, np.linspace(0, 1, 6))
    print(f"{'离散度分箱':22s}{'样本数':>8s}{'平均离散度':>12s}{'RMSE':>10s}")
    prev_rmse = None
    mono = True
    for i in range(5):
        m = (sp >= qs[i]) & (sp <= qs[i + 1] if i == 4 else sp < qs[i + 1])
        if m.sum() < 10:
            continue
        rmse = float(np.sqrt((sk[m] ** 2).mean()))
        if prev_rmse is not None and rmse < prev_rmse:
            mono = False
        prev_rmse = rmse
        print(f"[{qs[i]:6.1f}, {qs[i+1]:6.1f}] km{'':5s}{int(m.sum()):>8d}"
              f"{sp[m].mean():>12.1f}{rmse:>10.1f}")
    print(f"\n  spread-skill {'单调上升 ✓（离散度有信息量）' if mono else '非单调 ✗（离散度不指示误差）'}")
    print(f"  离散度均值 {sp.mean():.1f} km vs 集合均值误差均值 {sk.mean():.1f} km"
          f"  （理想：两者接近）")

    print("\n" + "=" * 74)
    print("3) 覆盖率（观测是否落在 均值 ± α·离散度 内）")
    print("=" * 74)
    print(f"{'α':>5s}{'名义(Gauss)':>12s}{'实测覆盖':>10s}")
    for a in (1.0, 1.645, 1.96):
        cov = float((sk <= a * sp).mean())
        nom = {1.0: 0.683, 1.645: 0.90, 1.96: 0.95}[a]
        print(f"{a:>5.2f}{nom:>12.3f}{cov:>10.3f}")
    print("\n判读：覆盖率远低于名义值 = 集合过度自信（离散度太小）；"
          "\n      远高于名义值 = 离散度太大。两种都说明需要调 eps。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
