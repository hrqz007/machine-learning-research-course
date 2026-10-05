"""Regenerate the one predeclared synthetic draw; never search seeds."""
from pathlib import Path
import csv, json, argparse
import numpy as np
ROOT = Path(__file__).resolve().parent

def generate(destination):
    p = json.loads((ROOT / 'data/protocol.json').read_text())
    rng = np.random.default_rng(p['seed'])
    n = p['n_train'] + p['n_test']
    x = rng.uniform(*p['x_uniform'], size=n)
    exposure = rng.uniform(*p['exposure_uniform'], size=n)
    mu = exposure * np.exp(p['true_beta'][0] + p['true_beta'][1] * x)
    y_p = rng.poisson(mu)
    frailty = rng.gamma(p['gamma_shape'], 1 / p['gamma_shape'], size=n)
    y_over = rng.poisson(mu * frailty)
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError('Choose a new output file; archived data are never overwritten.')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('w', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['row_id', 'split', 'x', 'exposure', 'true_mean', 'poisson_y', 'overdispersed_y'])
        for i in range(n):
            w.writerow([i + 1, 'train' if i < p['n_train'] else 'test',
                        format(x[i], '.17g'), format(exposure[i], '.17g'), format(mu[i], '.17g'),
                        int(y_p[i]), int(y_over[i])])
    return destination

if __name__ == '__main__':
    a = argparse.ArgumentParser(); a.add_argument('--out', required=True)
    print(generate(a.parse_args().out))
