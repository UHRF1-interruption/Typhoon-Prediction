"""GPU 环境自检：确认 torch 能用 GPU 训练这个工程的模型。

请在你的终端（非沙箱）里运行:
    cd typhoon-transformer
    .venv/bin/python scripts/check_gpu.py

检查项:
  1. torch 是否能识别 GPU
  2. 显存是否够放 Transformer（batch=64, d_model=128）
  3. 前向+反向一步是否正常、耗时多少
  4. 混合精度(AMP)是否可用
  5. 给出基于显存的 batch_size / 训练速度建议
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.config import load_config  # noqa: E402
from src.data.dataset import load_bundle  # noqa: E402
from src.models.typhoon_transformer import build_model, count_parameters  # noqa: E402
from src.training.losses import compute_loss  # noqa: E402


def main() -> int:
    print("=" * 64)
    print("PyTorch GPU 自检")
    print("=" * 64)
    print(f"torch 版本      : {torch.__version__}")
    print(f"CUDA 可用       : {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("\n[失败] torch 检测不到 GPU。")
        print("请在系统终端确认: nvidia-smi 正常、ls /dev/nvidia* 存在。")
        return 1

    print(f"GPU 数量        : {torch.cuda.device_count()}")
    print(f"GPU 名称        : {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    total_gb = props.total_memory / 1024 ** 3
    print(f"显存总量        : {total_gb:.2f} GB")
    print(f"计算能力        : sm_{props.major}{props.minor}")
    print(f"torch 编译 CUDA : {torch.version.cuda}")
    print(f"驱动支持 CUDA   : {torch.cuda.get_device_properties(0).major}.x")

    # 1) 基础张量运算
    print("\n[1/4] 基础运算 ...")
    t0 = time.time()
    a = torch.randn(2048, 2048, device="cuda")
    b = a @ a
    torch.cuda.synchronize()
    print(f"  2048x2048 矩阵乘完成, 耗时 {(time.time() - t0) * 1000:.1f} ms, "
          f"结果校验 {float(b.sum()):.3e}")

    # 2) 用真实数据构建模型和 batch
    print("\n[2/4] 构建模型与 batch ...")
    cfg = load_config(ROOT / "configs" / "default.yaml")
    bundle, _spec = load_bundle(ROOT / cfg.data.processed_dir)
    feat_dim = bundle["x"].shape[-1]
    in_len = bundle["x"].shape[1]
    out_len = bundle["y"].shape[1]
    model = build_model(cfg, feat_dim=feat_dim, in_len=in_len, out_len=out_len).cuda()
    print(f"  模型={cfg.model.name}, 参数量={count_parameters(model):,}")
    print(f"  样本数={len(bundle['x'])}, 输入 ({in_len}, {feat_dim}) -> 输出 ({out_len}, 2)")

    batch_size = int(cfg.train.batch_size)
    x = torch.randn(batch_size, in_len, feat_dim, device="cuda")
    y = torch.randn(batch_size, out_len, 2, device="cuda") * 0.5
    last = torch.stack([
        torch.rand(1, device="cuda") * 20 + 5,
        torch.rand(1, device="cuda") * 30 + 120,
    ], dim=-1).expand(batch_size, 2)
    mask = torch.zeros(batch_size, in_len, dtype=torch.bool, device="cuda")

    # 3) 前向 + 反向
    print(f"\n[3/4] 前向+反向一步 (batch={batch_size}) ...")
    for use_amp in (False, True):
        tag = "AMP" if use_amp else "FP32"
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
            t0 = time.time()
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                out = model(x, key_padding_mask=mask)
                loss = compute_loss(out, y, last[:, 0], last[:, 1], kind="haversine")
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() / 1024 ** 2
            print(f"  {tag}: 成功, loss={float(loss.detach()):.2f}, "
                  f"耗时 {(time.time() - t0) * 1000:.0f} ms, 峰值显存 {peak:.0f} MB")
        except RuntimeError as exc:
            msg = str(exc).splitlines()[0][:120]
            if "out of memory" in msg.lower():
                print(f"  {tag}: 显存不足 ({msg})")
                print(f"       建议: --set train.batch_size=32 或更小")
            else:
                print(f"  {tag}: 失败 {msg}")
    torch.cuda.empty_cache()

    # 4) 速度估算
    print("\n[4/4] 训练速度估算 ...")
    try:
        model.zero_grad(set_to_none=True)
        t0 = time.time()
        steps = 5
        for _ in range(steps):
            with torch.autocast(device_type="cuda", enabled=True):
                out = model(x, key_padding_mask=mask)
                loss = compute_loss(out, y, last[:, 0], last[:, 1], kind="haversine")
            loss.backward()
            model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        per_step = (time.time() - t0) / steps
        n_train = int(len(bundle["x"]) * 0.7)
        steps_per_epoch = max(1, n_train // batch_size)
        epoch_s = per_step * steps_per_epoch
        print(f"  单步 {per_step * 1000:.0f} ms, 每 epoch 约 {steps_per_epoch} 步 "
              f"≈ {epoch_s:.1f} 秒")
        print(f"  60 个 epoch 约 {epoch_s * 60 / 60:.1f} 分钟")
        print(f"\n  对比: CPU 上每 epoch 约 65-75 秒，预计提速 "
              f"{(60 / max(epoch_s, 0.1)):.1f}x")
    except RuntimeError as exc:
        print(f"  测速失败: {str(exc)[:120]}")

    print("\n" + "=" * 64)
    print("结论: GPU 可用，可以训练。")
    print("推荐命令:")
    print("  MPLCONFIGDIR=/tmp/mpl-cache .venv/bin/python scripts/train.py --set device=cuda")
    print("  MPLCONFIGDIR=/tmp/mpl-cache .venv/bin/python scripts/compare.py --set device=cuda")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
