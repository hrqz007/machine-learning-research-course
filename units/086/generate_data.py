"""Generate original XOR Gaussian-corner data, without network access."""
from pathlib import Path
import argparse, hashlib, json
import numpy as np

def generate(directory):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(85086)
    splits = {}
    for name, n in [('train', 323), ('validation', 157), ('test', 160)]:
        bits = rng.integers(0, 2, size=(n, 2))
        X = 2 * bits - 1 + rng.normal(0, .32, size=(n, 2))
        y = np.logical_xor(bits[:, 0], bits[:, 1]).astype(int)
        splits[name] = dict(X=X.tolist(), y=y.tolist())
    obj = dict(seed=85086, generator='independent binary corners + N(0,0.32^2 I)', splits=splits)
    text = json.dumps(obj, ensure_ascii=False, indent=2) + '\n'
    (directory / 'dataset.json').write_text(text)
    manifest = dict(seed=85086, sha256=hashlib.sha256(text.encode()).hexdigest(),
                    sizes={k: len(v['y']) for k,v in splits.items()},
                    provenance='Original educational synthetic data; no personal data or external downloads',
                    license='CC0-1.0', split_policy='Generate disjoint independent samples before training')
    (directory / 'generation.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest
if __name__ == '__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--directory',default='outputs/generated-data')
    print(json.dumps(generate(ap.parse_args().directory), indent=2))
