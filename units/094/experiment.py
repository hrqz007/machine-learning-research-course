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


def contrastive(z, tau=0.2):
    """First B rows and last B rows are paired views; self-similarity is excluded."""
    if tau <= 0 or len(z) % 2:
        raise ValueError("positive tau and an even number of views required")
    # Unit-length rows make dot products cosine similarities.
    z = F.normalize(z, dim=1)
    n = len(z)
    b = n // 2
    # Each row is one classification problem over the other views.
    logits = z @ z.T / tau
    # A view cannot select itself as its positive.
    logits = logits.masked_fill(torch.eye(n, dtype=torch.bool), float("-inf"))
    # Pair first-half row i with second-half row i+B, and vice versa.
    target = (torch.arange(n) + b) % n
    return F.cross_entropy(logits, target)


def main():
    started = time.perf_counter()
    torch.manual_seed(94)
    x, y = images(128, 940, 0.18)
    test, ty = images(200, 941, 0.18)
    np.savez_compressed(
        ROOT / "data/images.npz",
        unlabeled=x.numpy(),
        probe_y=y.numpy(),
        test=test.numpy(),
        test_y=ty.numpy(),
    )
    encoder = Encoder()
    initial = copy.deepcopy(encoder)
    projector = nn.Linear(8, 8)
    opt = torch.optim.Adam(
        list(encoder.parameters()) + list(projector.parameters()), lr=0.01
    )
    curve = []
    for step in range(180):
        # Independent noise is label-preserving for this synthetic direction task at this scale.
        # Both views derive from training images; test images never enter this objective.
        a = x + 0.08 * torch.randn_like(x)
        b = x + 0.08 * torch.randn_like(x)
        # 清除上一步梯度；PyTorch默认会累加，所以每次独立更新前必须清零。
        opt.zero_grad(set_to_none=True)
        loss = contrastive(projector(encoder(torch.cat([a, b]))))
        # 反向传播沿实际计算图计算所有可训练参数的梯度。
        loss.backward()
        # 只有此处真正修改参数；记录loss时detach避免保留整条计算图。
        opt.step()
        curve.append(float(loss.detach()))
    results = {}
    for name, enc in [("random", initial), ("contrastive", encoder)]:
        with torch.no_grad():
            features = enc(x)
            tfeatures = enc(test)
        torch.manual_seed(0)
        head = nn.Linear(8, 2)
        train(head, features, y, 150, 0.04)
        results[name] = {
            "linear_probe_accuracy": accuracy(head, tfeatures, ty),
            "feature_std_mean": float(features.std(0).mean()),
        }
    collapsed = torch.ones(16, 8)
    collapse_loss = float(contrastive(collapsed))
    plt.figure(figsize=(7, 3.5))
    plt.plot(curve)
    plt.xlabel("Unlabeled optimizer updates")
    plt.ylabel("InfoNCE loss")
    saveplot("01_contrastive.png")
    with torch.no_grad():
        h = encoder(x).numpy()
    # SVD is visualization only, fitted to training features; it is not used by the probe.
    h = h - h.mean(0)
    u, s, v = np.linalg.svd(h, full_matrices=False)
    p = h @ v[:2].T
    plt.figure(figsize=(6, 4))
    plt.scatter(p[:, 0], p[:, 1], c=y.numpy(), cmap="coolwarm", s=16)
    plt.xlabel("Training PCA 1")
    plt.ylabel("Training PCA 2")
    saveplot("02_features.png")
    save(
        {
            "probe": results,
            "initial_loss": curve[0],
            "final_loss": curve[-1],
            "collapsed_loss_16_views": collapse_loss,
            "collapsed_expected_log15": float(np.log(15)),
            "unlabeled_updates": 180,
            "probe_updates": 150,
            "seconds": time.perf_counter() - started,
        }
    )


if __name__ == "__main__":
    main()
