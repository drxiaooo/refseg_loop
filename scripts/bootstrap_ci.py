import argparse
import numpy as np


def bootstrap_ci(x, B=5000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=np.float32)
    n = len(x)

    means = np.empty(B, dtype=np.float32)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        means[b] = x[idx].mean()

    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return float(x.mean()), lo, hi


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--path", type=str, required=True)
    p.add_argument("--B", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    x = np.load(args.path)
    m, lo, hi = bootstrap_ci(x, B=args.B, seed=args.seed)

    print(args.path)
    print(f"mean={m:.4f}, 95% CI=[{lo:.4f}, {hi:.4f}]  (B={args.B}, n={len(x)})")


if __name__ == "__main__":
    main()
