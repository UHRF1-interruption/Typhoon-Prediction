"""训练主循环：AMPC/混合精度、梯度裁剪、早停、检查点、日志。"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.dataset import TrackWindowDataset, collate
from ..models.typhoon_transformer import build_model, count_parameters
from .losses import absolute_from_delta, compute_loss, error_km


def resolve_device(pref: str = "auto") -> torch.device:
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("配置要求 cuda，但当前环境 torch.cuda.is_available()=False。"
                               "检查 NVIDIA 驱动是否加载（/dev/nvidia0 是否存在）。")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_loader(ds: TrackWindowDataset, batch_size: int, shuffle: bool,
                num_workers: int = 0, persistent: bool = False) -> DataLoader:
    """构建 DataLoader。

    GPU 加速要点:
      * pin_memory=True 让 H2D 拷贝走异步，配合 batch_to_device 的 non_blocking
      * num_workers>0 时 persistent_workers 复用进程，避免每个 epoch 重启开销
        （数据是小张量、CPU 预处理轻，worker 主要用来重叠 IO 与计算）
    """
    has_cuda = torch.cuda.is_available()
    nw = int(num_workers)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate,
        num_workers=nw, pin_memory=has_cuda,
        persistent_workers=bool(persistent and nw > 0),
        prefetch_factor=(4 if nw > 0 else None),
        drop_last=False,
    )


def batch_to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()}


def data_loss(pred_delta: torch.Tensor, batch: dict, loss_kind: str = "haversine",
              huber_delta_km: float = 100.0) -> torch.Tensor:
    """纯数据损失。评估/早停用它，保证与历史口径可比（不含物理项）。"""
    return compute_loss(pred_delta, batch["y_delta"], batch["last_abs"][:, 0],
                        batch["last_abs"][:, 1], kind=loss_kind,
                        huber_delta_km=huber_delta_km)


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device,
             interval_hours: float, loss_kind: str = "haversine") -> dict:
    """在给定 loader 上推理并返回指标。模型恒输出 delta。"""
    model.eval()
    all_pred_abs, all_true_abs = [], []
    total_loss, total_n = 0.0, 0
    for batch in loader:
        batch = batch_to_device(batch, device)
        pred = model(batch["x"], key_padding_mask=~batch["valid_mask"],
                     last_lat=batch["last_abs"][:, 0],
                     last_lon=batch["last_abs"][:, 1])
        # 统一成 delta 契约：pred_mode 若为 absolute 也在此转换
        if getattr(model, "pred_mode", "delta") == "absolute":
            pred_delta = pred - batch["last_abs"].unsqueeze(1)
            pred_delta[..., 1] = (pred_delta[..., 1] + 180) % 360 - 180
        else:
            pred_delta = pred

        loss = data_loss(pred_delta, batch, loss_kind)
        total_loss += loss.detach().item() * batch["x"].size(0)
        total_n += batch["x"].size(0)

        pred_abs = absolute_from_delta(pred_delta.cpu().numpy(),
                                       batch["last_abs"].cpu().numpy())
        all_pred_abs.append(pred_abs)
        all_true_abs.append(batch["y_abs"].cpu().numpy())

    if total_n == 0:
        return {"loss": float("nan"), "mae_by_step": np.array([]), "mae_overall": float("nan"),
                "horizon": np.array([])}
    pred_abs = np.concatenate(all_pred_abs)
    true_abs = np.concatenate(all_true_abs)
    dist = error_km(pred_abs, true_abs)
    mae_by_step = dist.mean(axis=0)
    return {
        "loss": total_loss / total_n,
        "mae_by_step": mae_by_step,
        "rmse_by_step": np.sqrt((dist ** 2).mean(axis=0)),
        "mae_overall": float(dist.mean()),
        "horizon": np.arange(1, dist.shape[1] + 1) * interval_hours,
        "pred_abs": pred_abs,
        "true_abs": true_abs,
    }


@torch.no_grad()
def predict(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict:
    """导出预测绝对坐标，供作图/深度评估。"""
    model.eval()
    preds, trues, lasts, years = [], [], [], []
    for batch in loader:
        batch = batch_to_device(batch, device)
        pred = model(batch["x"], key_padding_mask=~batch["valid_mask"],
                     last_lat=batch["last_abs"][:, 0],
                     last_lon=batch["last_abs"][:, 1])
        if getattr(model, "pred_mode", "delta") == "absolute":
            delta = pred - batch["last_abs"].unsqueeze(1)
            delta[..., 1] = (delta[..., 1] + 180) % 360 - 180
        else:
            delta = pred
        preds.append(absolute_from_delta(delta.cpu().numpy(), batch["last_abs"].cpu().numpy()))
        trues.append(batch["y_abs"].cpu().numpy())
        lasts.append(batch["last_abs"].cpu().numpy())
        years.append(batch["year"].cpu().numpy())
    return {
        "pred_abs": np.concatenate(preds),
        "true_abs": np.concatenate(trues),
        "last_abs": np.concatenate(lasts),
        "year": np.concatenate(years),
    }


def build_model_for(cfg, bundle: dict):
    feat_dim = bundle["x"].shape[-1]
    in_len = bundle["x"].shape[1]
    out_len = bundle["y"].shape[1]
    return build_model(cfg, feat_dim=feat_dim, in_len=in_len, out_len=out_len)


def train(cfg, bundle: dict, spec, splits: dict[str, np.ndarray] | None = None) -> dict:
    """完整训练流程，返回最佳模型与历史。"""
    out_dir = Path(cfg.output.dir) / cfg.output.run_name
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)

    device = resolve_device(str(cfg.device))
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))

    if splits is None:
        from ..data.dataset import split_by_year
        splits = split_by_year(bundle, list(cfg.data.val_years), list(cfg.data.test_years))

    train_ds = TrackWindowDataset(bundle, spec, splits["train"])
    val_ds = TrackWindowDataset(bundle, spec, splits["val"])
    test_ds = TrackWindowDataset(bundle, spec, splits["test"])
    interval_hours = float(cfg.data.interval_hours)

    train_loader = make_loader(train_ds, int(cfg.train.batch_size), True,
                              int(cfg.train.num_workers), persistent=True)
    val_loader = make_loader(val_ds, int(cfg.train.batch_size) * 2, False, int(cfg.train.num_workers))
    test_loader = make_loader(test_ds, int(cfg.train.batch_size) * 2, False, int(cfg.train.num_workers))

    model = build_model_for(cfg, bundle).to(device)
    n_params = count_parameters(model)
    print(f"[train] 模型={cfg.model.name}, 参数量={n_params:,}, 设备={device}")
    print(f"[train] 样本数 训练={len(train_ds)} 验证={len(val_ds)} 测试={len(test_ds)}")

    # 无标签物理约束（软惩罚）。与 §4.11 的硬约束不同：
    # 这里输出完全自由，惩罚只要求"解像物理系统"，不引用标签。
    law_w = float(getattr(cfg.train, "law_weight", 0.0) or 0.0)
    law_ctx = None
    k_param = None
    if law_w > 0.0:
        from .physics_laws import build_ctx
        law_ctx = build_ctx(spec.names, spec.mean, spec.std,
                            float(cfg.data.interval_hours))
        if law_ctx.has_steering:
            # 引导系数 k **由训练集闭式定标**并固定，不在训练中自由学习：
            # 自由学习的退化解是 k→0（等于"不要动"），实测 3 epoch 就压到 0.005。
            from .physics_laws import fit_steer_k
            _tr_idx = splits["train"] if splits is not None else np.arange(len(bundle["x"]))
            _x = torch.tensor(bundle["x"][_tr_idx], dtype=torch.float32)
            _x = (_x - torch.tensor(spec.mean, dtype=torch.float32)) \
                / torch.tensor(spec.std, dtype=torch.float32)
            _y = torch.tensor(bundle["y"][_tr_idx][:, 0], dtype=torch.float32)
            _lat = torch.tensor(bundle["last_abs"][_tr_idx, 0], dtype=torch.float32)
            k_val, k_r2 = fit_steer_k(_y, _x, _lat, law_ctx)
            k_param = torch.nn.Parameter(torch.tensor(float(k_val)), requires_grad=False)
            print(f"[laws] 启用无标签物理惩罚：steering + smooth + curvature"
                  f"（k 由训练集定标 = {k_val:.3f}，R² = {k_r2:.3f}）")
        else:
            print("[laws] 启用无标签物理惩罚：smooth + curvature"
                  "（输入无 u500/v500，跳过 steering）")

    trainable = [p for p in model.parameters() if p.requires_grad]
    # k 是定标常数、不参与优化，故不加入 trainable
    skip_training = len(trainable) == 0      # 例如持续法基线：无参数，直接评估
    if skip_training:
        print(f"[train] 模型={cfg.model.name} 无训练参数，跳过训练直接评估")

    optimizer = None
    if not skip_training:
        optimizer = torch.optim.AdamW(trainable, lr=float(cfg.train.lr),
                                      weight_decay=float(cfg.train.weight_decay))
    epochs = int(cfg.train.epochs)
    warmup = int(cfg.train.warmup_epochs)
    scheduler = None
    if optimizer is not None:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda e: min(1.0, (e + 1) / max(1, warmup)) *
                      (0.5 * (1 + math.cos(math.pi * max(0, e - warmup) / max(1, epochs - warmup)))),
        )
    use_amp = bool(cfg.train.amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None

    # 物理项自适应权重（可选）。WHC-PINN 指出硬约束会带来梯度失衡；
    # 固定权重靠试凑，这里用 GradNorm 让四项对共享参数的梯度范数趋同。
    # 代价：每 balance_every 步多做 4 次反传，默认 50 步一次，开销很小。
    balancer = None
    balance_every = int(getattr(cfg.train, "balance_every", 50) or 50)
    if bool(getattr(cfg.train, "physics_adaptive", False)) and \
            float(getattr(cfg.train, "physics_weight", 0.0) or 0.0) > 0.0:
        from .physics import GradNormBalancer
        balancer = GradNormBalancer(["velocity", "accel", "turn", "endpoint"])
        print("[physics] 启用 GradNorm 自适应权重")

    best_val = float("inf")
    best_epoch = -1
    bad_epochs = 0
    history = []
    ckpt_path = out_dir / "checkpoints" / cfg.output.ckpt_name

    for epoch in range(0 if skip_training else epochs):
        model.train()
        t0 = time.time()
        running, seen = 0.0, 0
        for step, batch in enumerate(train_loader):
            batch = batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                pred = model(batch["x"], key_padding_mask=~batch["valid_mask"],
                     last_lat=batch["last_abs"][:, 0],
                     last_lon=batch["last_abs"][:, 1])
                if getattr(model, "pred_mode", "delta") == "absolute":
                    delta = pred - batch["last_abs"].unsqueeze(1)
                    delta[..., 1] = (delta[..., 1] + 180) % 360 - 180
                else:
                    delta = pred
                phys_w_token = 0.0
                data_l = compute_loss(delta, batch["y_delta"], batch["last_abs"][:, 0],
                                      batch["last_abs"][:, 1], kind=str(cfg.train.loss),
                                      huber_delta_km=float(cfg.model.huber_delta_km))
                loss = data_l
                phys_w = float(getattr(cfg.train, "physics_weight", 0.0) or 0.0)
                phys_w_token = phys_w
                if phys_w > 0.0:
                    from .physics import (physics_loss, saturation_rate)
                    if balancer is not None and (step + 1) % balance_every == 0:
                        # 单独算四项并按其梯度范数更新权重（不参与本次反传）
                        from .physics import physics_terms as _pt
                        _raw = _pt(delta, batch["y_delta"],
                                   batch["last_abs"][:, 0], batch["last_abs"][:, 1])
                        from .physics import (SCALE_VEL_KM, SCALE_ACC_KM,
                                              SCALE_TURN_DEG, SCALE_ENDPOINT_KM)
                        _norm = {
                            "velocity": _raw["velocity"] / SCALE_VEL_KM,
                            "accel": _raw["accel"] / SCALE_ACC_KM,
                            "turn": _raw["turn"] / SCALE_TURN_DEG,
                            "endpoint": _raw["endpoint"] / SCALE_ENDPOINT_KM,
                        }
                        _params = [q for q in model.parameters() if q.requires_grad]
                        if _params:
                            balancer.update(_norm, _params)
                    w_terms = balancer.w if balancer is not None else None
                    pl, pterms = physics_loss(
                        delta, batch["y_delta"],
                        batch["last_abs"][:, 0], batch["last_abs"][:, 1],
                        weights=w_terms)
                    if not torch.isfinite(pl):
                        pl = torch.zeros((), device=delta.device)
                    # 权重必须相对于**数据损失的当前量级**，否则形同虚设：
                    # 数据损失是 km 量级(~100)，物理项已归一化成 O(1)(~3.6)，
                    # 若直接写 loss + w*pl，w=0.1 时物理项只占 0.36%，
                    # 实测 w=0.01/0.03/0.1/0.3 的结果完全相同（都等于没加）。
                    # 这正是 Rohrhofer et al. 2023 指出的"量纲缩放造成失衡"。
                    # 改为乘以 data_l.detach() 后，w 表示"物理项占数据项的比例"。
                    loss = loss + phys_w * data_l.detach() * pl
                    if getattr(cfg.train, "log_physics", False) and step == 0:
                        pr = {k: round(float(v), 4) for k, v in pterms.items()}
                        # 饱和率是"硬约束把梯度堵住"的直接证据，必须一起看
                        raw = getattr(model, "last_raw", None)
                        sat = f" 饱和率={saturation_rate(raw)*100:.1f}%" \
                            if raw is not None else ""
                        wtxt = ""
                        if balancer is not None:
                            wtxt = " 权重=" + str({k: round(v, 2) for k, v
                                                   in balancer.w.items()})
                        print(f"  [physics] epoch {epoch} w={phys_w:.3g} "
                              f"{pr}{sat}{wtxt}")

                # 无标签物理惩罚：与上面的区别是它**不引用标签**
                if law_w > 0.0 and law_ctx is not None:
                    from .physics_laws import law_loss
                    ll, lterms, _k = law_loss(delta, batch["x"],
                                              batch["last_abs"][:, 0], law_ctx,
                                              k=k_param)
                    if torch.isfinite(ll):
                        # 同样按相对量级加权（数据损失是 km，惩罚项已归一化成 O(1)）
                        loss = loss + law_w * data_l.detach() * ll
                        if getattr(cfg.train, "log_physics", False) and step == 0:
                            lterm_s = {k: round(float(v), 4)
                                       for k, v in lterms.items()}
                            kv = f"  k={float(k_param):.3f}" \
                                if k_param is not None else ""
                            print(f"  [laws] epoch {epoch} w={law_w:.3g} "
                                  f"{lterm_s}{kv}")
            if not torch.isfinite(loss):
                print(f"[warn] epoch {epoch} step {step} 出现非有限损失，跳过该 batch")
                optimizer.zero_grad(set_to_none=True)
                continue
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.train.grad_clip))
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.train.grad_clip))
                optimizer.step()

            # 日志里报的是**数据损失**（km），不是含物理项的总损失——
            # 否则加了物理项后 "train=647 km" 这种数字不可解读。
            running += (data_l if phys_w_token > 0.0 else loss).detach().item() \
                * batch["x"].size(0)
            seen += batch["x"].size(0)
            if step % int(cfg.train.log_interval) == 0:
                print(f"  epoch {epoch} step {step}/{len(train_loader)} loss={running / seen:.2f} km")

        scheduler.step()
        train_loss = running / max(1, seen)
        val_metrics = evaluate(model, val_loader, device, interval_hours, str(cfg.train.loss)) if len(val_ds) else None
        val_mae = val_metrics["mae_overall"] if val_metrics else float("nan")
        record = {
            "epoch": epoch, "train_loss_km": train_loss, "val_loss": val_metrics["loss"] if val_metrics else None,
            "val_mae_km": val_mae, "lr": optimizer.param_groups[0]["lr"],
            "seconds": round(time.time() - t0, 1),
        }
        history.append(record)
        kmsg = f" k={float(k_param):.3f}" if k_param is not None else ""
        print(f"[epoch {epoch}] train={train_loss:.2f}km val_mae={val_mae:.2f}km "
              f"lr={record['lr']:.2e} {record['seconds']}s{kmsg}")

        score = val_mae if np.isfinite(val_mae) else train_loss
        if score < best_val - 1e-6:
            best_val, best_epoch, bad_epochs = score, epoch, 0
            torch.save({"model": model.state_dict(), "cfg": cfg.to_dict(),
                        "feat_dim": bundle["x"].shape[-1], "in_len": bundle["x"].shape[1],
                        "out_len": bundle["y"].shape[1], "epoch": epoch,
                        "val_mae_km": best_val}, ckpt_path)
        else:
            bad_epochs += 1
            if bad_epochs >= int(cfg.train.early_stop_patience):
                print(f"[train] 早停于 epoch {epoch}（最佳 epoch {best_epoch}）")
                break

    if skip_training and not ckpt_path.exists():
        best_epoch = 0
        torch.save({"model": model.state_dict(), "cfg": cfg.to_dict(),
                    "feat_dim": bundle["x"].shape[-1], "in_len": bundle["x"].shape[1],
                    "out_len": bundle["y"].shape[1], "epoch": 0, "val_mae_km": None},
                   ckpt_path)

    # 载入最佳权重做最终测试
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
    test_metrics = evaluate(model, test_loader, device, interval_hours, str(cfg.train.loss)) if len(test_ds) else None

    (out_dir / "logs" / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "best_epoch": best_epoch,
        "best_val_mae_km": best_val,
        "n_params": n_params,
        "model": str(cfg.model.name),
        "test": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                 for k, v in (test_metrics or {}).items() if k not in {"pred_abs", "true_abs"}},
    }
    (out_dir / "logs" / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "model": model,
        "best_epoch": best_epoch,
        "best_val_mae_km": best_val,
        "test_metrics": test_metrics,
        "history": history,
        "device": device,
        "pred": predict(model, test_loader, device) if len(test_ds) else None,
        "out_dir": out_dir,
    }
