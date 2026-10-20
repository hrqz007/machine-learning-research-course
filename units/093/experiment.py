"""CPU-only, locally generated educational experiments. No downloads or services."""

from pathlib import Path
import json, time, copy, hashlib, platform
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
# One CPU thread makes timing more interpretable and avoids oversubscribing laptops.
torch.set_num_threads(1)


def images(n, seed, noise=0.20):
    """Balanced orientations, random positions, independent Gaussian pixel noise."""
    rng = np.random.default_rng(seed)
    y = np.arange(n) % 2
    x = rng.normal(0, noise, (n, 1, 10, 10)).astype("float32")
    for k, c in enumerate(y):
        row, col = rng.integers(2, 7, size=2)
        if c == 0:
            x[k, 0, row, col - 2 : col + 3] += 1
        else:
            x[k, 0, row - 2 : row + 3, col] += 1
    return torch.tensor(x), torch.tensor(y, dtype=torch.long)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 6, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(6, 8, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

    def forward(self, x):
        return self.net(x)


class Vision(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.head = nn.Linear(8, 2)

    def forward(self, x):
        return self.head(self.encoder(x))


def train(model, x, y, steps=100, lr=0.02):
    """Full-batch Adam: steps count actual updates, not epochs with hidden batches."""
    model.train()
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    losses = []
    for step in range(steps):
        # 清除上一步梯度；PyTorch默认会累加，所以每次独立更新前必须清零。
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x), y)
        # 反向传播沿实际计算图计算所有可训练参数的梯度。
        loss.backward()
        # 只有此处真正修改参数；记录loss时detach避免保留整条计算图。
        opt.step()
        losses.append(float(loss.detach()))
    return losses


def accuracy(model, x, y):
    # 推理模式和禁止梯度是两个独立开关，评价时一起使用。
    model.eval()
    with torch.no_grad():
        return float((model(x).argmax(1) == y).float().mean())


def save(result):
    result["environment"] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": "cpu",
        "threads": 1,
    }
    (ROOT / "outputs/results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2)
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def saveplot(name):
    plt.tight_layout()
    plt.savefig(ROOT / "figures" / name, dpi=160)
    plt.close()


def iou(a, b):
    """Half-open continuous boxes (left,top,right,bottom); no pixel '+1'."""
    if min(a[2] - a[0], a[3] - a[1], b[2] - b[0], b[3] - b[1]) <= 0:
        raise ValueError("positive-area boxes required")
    inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )
    return inter / (
        (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    )


def audit(a, b):
    """Exact hashes plus nearest raw-image RMSE. Threshold is a screening rule only."""
    aa = a.numpy().reshape(len(a), -1)
    bb = b.numpy().reshape(len(b), -1)
    ha = {hashlib.sha256(v.tobytes()).hexdigest() for v in aa}
    exact = sum(hashlib.sha256(v.tobytes()).hexdigest() in ha for v in bb)
    nearest = np.sqrt(((aa[:, None, :] - bb[None, :, :]) ** 2).mean(2)).min(0)
    return {
        "exact_count": exact,
        "near_count_rmse_lt_0_03": int((nearest < 0.03).sum()),
        "minimum_rmse": float(nearest.min()),
    }


def main():
    started = time.perf_counter()
    torch.manual_seed(93)
    source, sy = images(256, 930, noise=0.12)
    x, y = images(40, 931, noise=0.32)
    test, ty = images(200, 932, noise=0.32)
    np.savez_compressed(
        ROOT / "data/images.npz",
        source=source.numpy(),
        source_y=sy.numpy(),
        train=x.numpy(),
        train_y=y.numpy(),
        test=test.numpy(),
        test_y=ty.numpy(),
    )
    pre = Vision()
    source_curve = train(pre, source, sy, 120)
    records = []
    curves = {}
    for seed in [0, 1, 2]:
        for mode in ["scratch", "frozen", "finetune"]:
            torch.manual_seed(seed)
            m = Vision()
            if mode != "scratch":
                m.encoder.load_state_dict(pre.encoder.state_dict())
            # All target heads have the same initial seed; only encoder treatment differs.
            if mode == "frozen":
                for p in m.encoder.parameters():
                    p.requires_grad_(False)
            before = [p.detach().clone() for p in m.encoder.parameters()]
            curve = train(m, x, y, 100, lr=0.02 if mode != "finetune" else 0.005)
            delta = max(
                float((a - b).abs().max())
                for a, b in zip(before, m.encoder.parameters())
            )
            records.append(
                {
                    "seed": seed,
                    "mode": mode,
                    "accuracy": accuracy(m, test, ty),
                    "encoder_max_change": delta,
                    "target_updates": 100,
                }
            )
            if seed == 0:
                curves[mode] = curve
    # An injected duplicate pair must be caught. Original train/test arrays remain unmodified.
    injected = torch.cat([test, x[:1], x[:1] + 0.001], dim=0)
    metric = {
        "box_iou": iou([0, 0, 2, 2], [1, 0, 3, 2]),
        "detection_tp_fp_fn_at_0_5": [1, 1, 1],
        "foreground_iou": 3 / 5,
        "foreground_dice": 6 / 8,
        "pixel_accuracy": 14 / 16,
    }
    plt.figure(figsize=(7, 3.5))
    for k, v in curves.items():
        plt.plot(v, label=k)
    plt.xlabel("Target optimizer updates")
    plt.ylabel("Cross entropy")
    plt.legend()
    saveplot("01_transfer.png")
    fig, axes = plt.subplots(2, 5, figsize=(8, 3.5))
    for ax, k in zip(axes.flat, range(10)):
        ax.imshow(x[k, 0], cmap="gray")
        ax.set_title("horizontal" if y[k] == 0 else "vertical")
        ax.axis("off")
    saveplot("02_data.png")
    save(
        {
            "records": records,
            "source_updates": 120,
            "source_train_accuracy": accuracy(pre, source, sy),
            "source_test_audit": audit(source, test),
            "source_target_train_audit": audit(source, x),
            "original_audit": audit(x, test),
            "injected_audit": audit(x, injected),
            "metric_handcheck": metric,
            "seconds": time.perf_counter() - started,
        }
    )


if __name__ == "__main__":
    main()
