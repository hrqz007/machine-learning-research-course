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


def manual_loss_grad(w, x, y):
    """One hidden tanh unit and fixed output weights; explicit chain-rule gradient."""
    h = np.tanh(x @ w)
    pred = h
    error = pred - y
    loss = np.mean(error**2) / 2
    grad = x.T @ (error * (1 - h * h)) / len(x)
    return loss, grad


def gradcheck():
    x = np.array([[0.2, -0.3], [0.5, 0.1], [-0.4, 0.7]])
    y = np.array([0.1, 0.3, -0.2])
    w = np.array([0.4, -0.2])
    loss, g = manual_loss_grad(w, x, y)
    fd = np.zeros_like(w)
    for k in range(len(w)):
        shift = np.zeros_like(w)
        shift[k] = 1e-6
        fd[k] = (
            manual_loss_grad(w + shift, x, y)[0] - manual_loss_grad(w - shift, x, y)[0]
        ) / 2e-6
    tx = torch.tensor(x, dtype=torch.float64)
    tw = torch.tensor(w, dtype=torch.float64, requires_grad=True)
    tl = ((torch.tanh(tx @ tw) - torch.tensor(y)) ** 2).mean() / 2
    tl.backward()
    return {
        "finite_difference_max_error": float(abs(g - fd).max()),
        "autograd_max_error": float(abs(g - tw.grad.numpy()).max()),
    }


def main():
    started = time.perf_counter()
    x, y = images(128, 950, 0.3)
    test, ty = images(300, 951, 0.3)
    np.savez_compressed(
        ROOT / "data/images.npz",
        train=x.numpy(),
        train_y=y.numpy(),
        test=test.numpy(),
        test_y=ty.numpy(),
    )
    rows = []
    predictions = {}
    curves = {}
    all_curves = {}
    for seed in range(5):
        for mode in ["linear", "mlp", "cnn"]:
            torch.manual_seed(seed)
            if mode == "linear":
                m = nn.Sequential(nn.Flatten(), nn.Linear(100, 2))
            elif mode == "mlp":
                m = nn.Sequential(
                    nn.Flatten(), nn.Linear(100, 16), nn.Tanh(), nn.Linear(16, 2)
                )
            else:
                m = Vision()
            t = time.perf_counter()
            curve = train(m, x, y, 100, 0.02)
            elapsed = time.perf_counter() - t
            m.eval()
            with torch.no_grad():
                pred = m(test).argmax(1)
            predictions[f"{mode}_{seed}"] = pred.numpy()
            rows.append(
                {
                    "seed": seed,
                    "model": mode,
                    "accuracy": float((pred == ty).float().mean()),
                    "class0_accuracy": float((pred[ty == 0] == 0).float().mean()),
                    "class1_accuracy": float((pred[ty == 1] == 1).float().mean()),
                    "parameters": sum(p.numel() for p in m.parameters()),
                    "updates": 100,
                    "train_seconds": elapsed,
                    "error_indices": torch.where(pred != ty)[0].tolist(),
                }
            )
            all_curves[f"{mode}_{seed}"] = curve
            if seed == 0:
                curves[mode] = curve
    (ROOT / "outputs/training_curves.json").write_text(json.dumps(all_curves, indent=2))
    # Pair models by seed. Bootstrap independent training seeds, not correlated epochs.
    diff = np.array(
        [
            next(r["accuracy"] for r in rows if r["model"] == "cnn" and r["seed"] == s)
            - next(
                r["accuracy"] for r in rows if r["model"] == "mlp" and r["seed"] == s
            )
            for s in range(5)
        ]
    )
    rng = np.random.default_rng(955)
    boots = rng.choice(diff, (4000, 5), replace=True).mean(1)
    summary = {
        "paired_cnn_minus_mlp": diff.tolist(),
        "mean_difference": float(diff.mean()),
        "seed_bootstrap_percentile_95": np.quantile(boots, [0.025, 0.975]).tolist(),
        "limitation": "5 training seeds; shared fixed test set; not population-level uncertainty",
    }
    np.savez_compressed(ROOT / "outputs/predictions.npz", **predictions)
    plt.figure(figsize=(7, 3.5))
    for name in ["linear", "mlp", "cnn"]:
        vals = [r["accuracy"] for r in rows if r["model"] == name]
        plt.scatter([name] * 5, vals)
        plt.plot([name], [np.mean(vals)], "kx", markersize=12)
    plt.ylim(0, 1.05)
    plt.ylabel("Fixed-test accuracy across 5 seeds")
    saveplot("01_comparison.png")
    plt.figure(figsize=(7, 3.5))
    for name, c in curves.items():
        plt.plot(c, label=name)
    plt.xlabel("Optimizer updates")
    plt.ylabel("Training loss, seed 0")
    plt.legend()
    saveplot("02_training.png")
    save(
        {
            "gradient_check": gradcheck(),
            "runs": rows,
            "summary": summary,
            "seconds": time.perf_counter() - started,
        }
    )


if __name__ == "__main__":
    main()
