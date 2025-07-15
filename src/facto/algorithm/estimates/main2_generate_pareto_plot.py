from __future__ import annotations

import argparse
import math
import pathlib

from matplotlib import pyplot as plt

from facto.algorithm.estimates.main1_generate_cost_table_data import read_buckets


def save_cost_figure(*, csv_path: str | pathlib.Path, out_path: str | pathlib.Path | None, title: str | None = None, show: bool = False):
    buckets = read_buckets(csv_path)
    if title == "RSA IFP": # Note: Temporary solution.
      gid_costs = {
          1024: ([400_000_000], [3*1024 + math.ceil(0.002*1024*math.log2(1024))]),
          2048: ([2_700_000_000], [3*2048 + math.ceil(0.002*2048*math.log2(2048))]),
          3072: ([9_900_000_000], [3*3072 + math.ceil(0.002*3072*math.log2(3072))]),
      }
      chev_costs = {
          2048: ([2**40.87], [1730]),
          3072: ([2**42.7], [2415]),
          4096: ([2**43.49], [3096]),
          6144: ([2**45.36], [4430]),
          8192: ([2**46.63], [5781]),
      }
    k = 0
    fig: plt.Figure
    ax: plt.Axes
    fig, ax = plt.subplots()
    for n, v in sorted(buckets.items()):
        xs = []
        ys = []
        best_tofs = float('inf')
        for qubits, result in sorted(v.items()):
            tofs = result["tofs"]
            n = result["modulus_bitlength"]
            if tofs * 1.05 > best_tofs:
                continue
            best_tofs = tofs
            xs.append(tofs)
            ys.append(qubits)

        # Clip terrible qubit tradeoffs as Toffolis skyrocket at end of curve.
        ys = ys[::-1]
        xs = xs[::-1]
        while len(ys) > 2 and 0.997 < abs(ys[-1] / ys[-2]) < 1.003:
            xs.pop()
            ys.pop()
        ys = ys[::-1]
        xs = xs[::-1]

        ax.text(xs[0], ys[0], f'{n=}', horizontalalignment='left', verticalalignment='top', color=f'C{k}', fontsize=12)
        ax.plot(xs, ys, label=f'{n=}', zorder=100, marker='.', color=f'C{k}')
        if title == "RSA IFP": # Note: Temporary solution.
          if n in gid_costs:
              xs2, ys2 = gid_costs[n]
              ax.plot(xs2, ys2, zorder=100, marker='s', color=f'C{k}')
              ax.text(xs2[0], ys2[0], f'GE21:n={n}', horizontalalignment='left', verticalalignment='top', color=f'C{k}', fontsize=12)
          if n in chev_costs:
              xs2, ys2 = chev_costs[n]
              ax.plot(xs2, ys2, zorder=100, marker='p', color=f'C{k}')
              ax.text(xs2[0], ys2[0], f'CFS24:n={n}', horizontalalignment='left', verticalalignment='top', color=f'C{k}', fontsize=12)
        k += 1

    ax.loglog()
    ax.set_ylim(10**2, 10**4)
    ax.set_xlim(10**7, 10**15)
    ax.grid(zorder=1)
    if None != title:
        ax.set_title(title, fontsize=16)
    ax.set_xlabel(f'Toffoli Gates', fontsize=16)
    ax.set_ylabel(fr'$\mathbf{{Logical}}$ Qubits', fontsize=16)
    fig.set_dpi(200)
    fig.set_size_inches(12, 7)
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path)
        print(f'wrote file://{pathlib.Path(out_path).absolute()}')
    if show or out_path is None:
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', required=True, type=str)
    parser.add_argument('--out_path', default=None, type=str)
    parser.add_argument('--title', default=None, type=str)
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()

    csv_path = pathlib.Path(args.csv_path)
    if args.out_path is None:
        out_path = None
    else:
        out_path = pathlib.Path(args.out_path)
        out_path.parent.mkdir(exist_ok=True, parents=True)
    save_cost_figure(
        csv_path=csv_path,
        out_path=out_path,
        title=args.title,
        show=args.show,
    )


if __name__ == "__main__":
    main()
