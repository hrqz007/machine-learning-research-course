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


def make_corpus():
    # Original tiny artificial language: repeated templates make biases visible.
    sentences = []
    for _ in range(30):
        for animal in ["cat", "dog"]:
            sentences += [f"the {animal} eats food", f"the {animal} likes play"]
        for item in ["car", "bus"]:
            sentences += [f"the {item} uses road", f"the {item} moves fast"]
    sentences += ["rare comet crosses sky"] * 2
    return [s.split() for s in sentences]


def pairs(corpus, vocab):
    out = []
    for sent in corpus:
        ids = [vocab[w] for w in sent]
        for i, center in enumerate(ids):
            for j in range(max(0, i - 2), min(len(ids), i + 3)):
                if j != i:
                    out.append((center, ids[j]))
    return torch.tensor(out, dtype=torch.long)


def main():
    started = time.perf_counter()
    torch.manual_seed(96)
    corpus = make_corpus()
    vocab = {w: i for i, w in enumerate(sorted({w for s in corpus for w in s}))}
    ij = pairs(corpus, vocab)
    n = len(vocab)
    (ROOT / "data/corpus.json").write_text(json.dumps(corpus, indent=2))
    (ROOT / "data/vocabulary.json").write_text(json.dumps(vocab, indent=2))
    emb = nn.Embedding(n, 6)
    output = nn.Linear(6, n, bias=False)
    opt = torch.optim.Adam(list(emb.parameters()) + list(output.parameters()), lr=0.04)
    curve = []
    for step in range(180):
        # Full softmax predicts context from center. This is not negative-sampling word2vec.
        # 清除上一步梯度；PyTorch默认会累加，所以每次独立更新前必须清零。
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(output(emb(ij[:, 0])), ij[:, 1])
        # 反向传播沿实际计算图计算所有可训练参数的梯度。
        loss.backward()
        # 只有此处真正修改参数；记录loss时detach避免保留整条计算图。
        opt.step()
        curve.append(float(loss.detach()))
    ids = torch.tensor([vocab["cat"], vocab["cat"], vocab["bus"]])
    onehot = F.one_hot(ids, n).float()
    eq = float((emb(ids) - onehot @ emb.weight).abs().max())
    check = nn.Embedding(n, 6)
    check(ids).sum().backward()
    counts = torch.bincount(ids, minlength=n).float()[:, None].expand(n, 6)
    grad_error = float((check.weight.grad - counts).abs().max())
    weights = F.normalize(emb.weight.detach(), dim=1)
    sim = weights @ weights.T
    words = list(vocab)
    neighbors = {}
    for w in ["cat", "dog", "car", "bus", "comet"]:
        score = sim[vocab[w]].clone()
        score[vocab[w]] = -float("inf")
        neighbors[w] = [words[j] for j in score.topk(3).indices.tolist()]
    frequency = {w: sum(s.count(w) for s in corpus) for w in vocab}
    np.savez_compressed(
        ROOT / "outputs/embeddings.npz",
        input=emb.weight.detach().numpy(),
        output=output.weight.detach().numpy(),
    )
    plt.figure(figsize=(7, 3.5))
    plt.plot(curve)
    plt.xlabel("Full-batch updates")
    plt.ylabel("Context cross entropy")
    saveplot("01_loss.png")
    plt.figure(figsize=(8, 4))
    plt.bar(words, [frequency[w] for w in words])
    plt.xticks(rotation=60)
    plt.ylabel("Token frequency")
    saveplot("02_frequency.png")
    save(
        {
            "vocabulary_size": n,
            "sentences": len(corpus),
            "context_pairs": len(ij),
            "initial_loss": curve[0],
            "final_loss": curve[-1],
            "lookup_max_error": eq,
            "repeated_index_gradient_error": grad_error,
            "neighbors": neighbors,
            "frequency": frequency,
            "seconds": time.perf_counter() - started,
        }
    )


if __name__ == "__main__":
    main()
