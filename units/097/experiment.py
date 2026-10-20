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


def scalar_bptt(x, u, w):
    """Loss = h_T^2/2; h_0=0; return exact derivative with respect to shared w."""
    hs = [0.0]
    for v in x:
        hs.append(float(np.tanh(u * v + w * hs[-1])))
    adj = hs[-1]
    dw = 0.0
    for t in range(len(x), 0, -1):
        # Chain through tanh, then add this use of shared w to its total.
        da = adj * (1 - hs[t] ** 2)
        dw += da * hs[t - 1]
        # Propagate the remaining adjoint to the preceding hidden state.
        adj = da * w
    return 0.5 * hs[-1] ** 2, dw, hs


class Memory(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(1, 8)
        self.hidden = nn.Linear(8, 8, bias=False)
        self.head = nn.Linear(8, 2)

    def forward(self, x, initial=None, truncate=0, return_state=False):
        # Every independent sequence gets zero state unless a caller explicitly passes state.
        h = torch.zeros(len(x), 8) if initial is None else initial
        for t in range(x.shape[1]):
            if truncate and t and t % truncate == 0:
                h = h.detach()
            h = torch.tanh(self.input(x[:, t]) + self.hidden(h))
        logits = self.head(h)
        return (logits, h) if return_state else logits


def memory_data(n, length, seed):
    rng = np.random.default_rng(seed)
    y = np.arange(n) % 2
    x = rng.normal(0, 0.12, (n, length, 1)).astype("float32")
    x[:, 0, 0] = 2 * y - 1
    return torch.tensor(x), torch.tensor(y, dtype=torch.long)


def main():
    started = time.perf_counter()
    torch.manual_seed(97)
    x, y = memory_data(128, 6, 970)
    m = Memory()
    curve = train(m, x, y, 180, 0.015)
    np.savez_compressed(ROOT / "data/sequences.npz", train=x.numpy(), train_y=y.numpy())
    lengths = {}
    for length in [6, 12, 24, 48]:
        tx, ty = memory_data(256, length, 971 + length)
        lengths[str(length)] = accuracy(m, tx, ty)
    # Compare exact gradient to central finite difference and PyTorch on a short scalar RNN.
    xx = [0.2, -0.1, 0.4]
    u = 0.7
    w = 0.5
    loss, dw, hs = scalar_bptt(xx, u, w)
    eps = 1e-6
    fd = (scalar_bptt(xx, u, w + eps)[0] - scalar_bptt(xx, u, w - eps)[0]) / (2 * eps)
    tw = torch.tensor(w, dtype=torch.float64, requires_grad=True)
    h = torch.tensor(0.0, dtype=torch.float64)
    for v in xx:
        h = torch.tanh(u * v + tw * h)
    (0.5 * h * h).backward()
    tx, ty = memory_data(16, 6, 979)
    with torch.no_grad():
        expected = m(tx)
        _, state = m(x[:16], return_state=True)
        leaked = m(tx, initial=state)
    # Detaching preserves the forward computation but changes the gradient graph.
    full = copy.deepcopy(m)
    trunc = copy.deepcopy(m)
    full_logits = full(tx)
    trunc_logits = trunc(tx, truncate=2)
    F.cross_entropy(full_logits, ty).backward()
    F.cross_entropy(trunc_logits, ty).backward()
    grad_delta = float((full.hidden.weight.grad - trunc.hidden.weight.grad).abs().max())
    plt.figure(figsize=(7, 3.5))
    plt.plot(curve)
    plt.xlabel("Updates at training length 6")
    plt.ylabel("Cross entropy")
    saveplot("01_training.png")
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
    ax[0].plot([int(k) for k in lengths], list(lengths.values()), "o-")
    ax[0].set_xlabel("Sequence length")
    ax[0].set_ylabel("First-bit test accuracy")
    ax[0].set_ylim(0, 1.05)
    for gain in [0.5, 0.9, 1.1]:
        ax[1].semilogy(np.arange(1, 41), gain ** np.arange(1, 41), label=f"gain {gain}")
    ax[1].set_xlabel("Jacobian product length")
    ax[1].set_ylabel("Illustrative scalar gain")
    ax[1].legend()
    saveplot("02_memory.png")
    save(
        {
            "length_accuracy": lengths,
            "manual_hidden_states": hs,
            "manual_loss": loss,
            "manual_dw": dw,
            "finite_difference_error": abs(dw - fd),
            "autograd_error": abs(dw - float(tw.grad)),
            "reset_vs_leaked_max_logit_difference": float(
                (expected - leaked).abs().max()
            ),
            "truncate_forward_max_error": float(
                (full_logits - trunc_logits).abs().max()
            ),
            "truncate_gradient_max_difference": grad_delta,
            "seconds": time.perf_counter() - started,
        }
    )


if __name__ == "__main__":
    main()
