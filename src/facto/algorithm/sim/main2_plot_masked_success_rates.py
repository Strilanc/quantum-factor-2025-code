import argparse
import collections
import pathlib

import numpy as np
import sinter
from matplotlib import pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in', nargs='+', type=str, required=True)
    parser.add_argument('--out_path', type=str, required=True)
    parser.add_argument('--xmax', type=int, default=None)
    parser.add_argument('--title', type=str, default=None)
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()

    if args.out is None:
        out_path = None
    else:
        out_path = pathlib.Path(args.out)
    max_x = args.xmax
    title = args.title

    p2m2su: collections.defaultdict[float, collections.defaultdict[int, float]] = collections.defaultdict(lambda: collections.defaultdict(float))
    p2m2sh: collections.defaultdict[float, collections.defaultdict[int, int]] = collections.defaultdict(lambda: collections.defaultdict(int))
    moduli = set()

    for path in getattr(args, 'in'):
        with open(path) as f:
            for line in f.readlines():
                line = line.strip()
                if not line:
                    continue
                if line.startswith('modulus,'):
                    continue
                pieces = line.split(',')
                modulus, mask_proportion, shots, successes = pieces
                modulus = int(modulus)
                shots = int(shots)
                mask_proportion = float(mask_proportion)
                successes = float(successes)
                p2m2sh[mask_proportion][modulus] += shots
                p2m2su[mask_proportion][modulus] += successes
                moduli.add(modulus)

    ax: plt.Axes
    fig: plt.Figure
    fig, ax = plt.subplots()

    ps = set(p2m2sh.keys())
    k = 0
    moduli = sorted(moduli)
    if max_x is None:
        max_x = moduli[-1]
    for p in sorted(ps):
        if p:
            ax.hlines(1 - p, 0, max_x, linestyle='--', color=f'black', zorder=5)
            k += 1

    k = 0
    for p in sorted(ps):
        if not p:
            continue
        xs = []
        ys = []
        ys_low = []
        ys_high = []
        for modulus in moduli:
            if modulus > max_x + 30:
                continue
            shot_0 = p2m2sh[0].get(modulus, 0)
            succ_0 = p2m2su[0].get(modulus, 0)
            shot_p = p2m2sh[p].get(modulus, 0)
            succ_p = p2m2su[p].get(modulus, 0)
            if shot_p and shot_0:
                fit_0 = sinter.fit_binomial(num_shots=shot_0, num_hits=round(succ_0), max_likelihood_factor=100)
                fit_p = sinter.fit_binomial(num_shots=shot_p, num_hits=round(succ_p), max_likelihood_factor=100)
                xs.append(modulus)
                ys.append(fit_p.best / fit_0.best)
                ys_low.append(fit_p.low / fit_0.high)
                ys_high.append(fit_p.high / fit_0.low)
        if xs:
            ax.errorbar(xs, ys, (np.array(ys) - np.array(ys_low), np.array(ys_high) - np.array(ys)), linestyle='', marker=None, color=f'C{k}', alpha=0.3, zorder=4)
            ax.plot(xs, ys, label=f'Mask Proportion S={p}', color=f'C{k}', linestyle='', marker='.', zorder=3, markersize=3)
            k += 1

    ax.hlines(0, 0, 0, linestyle='--', color=f'black', label='1 - S')
    ax.set_ylabel(f"Masked/Unmasked Success Ratio", fontsize=16)
    ax.set_xlabel("Semiprime", fontsize=16)
    ax.set_ylim(0, 1.4)
    ax.set_xlim(0, max_x)
    if title is not None:
        ax.set_title(title, fontsize=20)
    ax.legend()
    fig.set_dpi(200)
    fig.set_size_inches(12, 7)
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path)
        print(f'wrote file://{pathlib.Path(out_path).absolute()}')
    if out_path is None or args.show:
        plt.show()


if __name__ == '__main__':
    main()
