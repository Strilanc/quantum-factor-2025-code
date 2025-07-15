from __future__ import annotations

import argparse
import pathlib
from typing import Any, Callable

from facto.algorithm.estimates._cost_config import CostConfig
from facto.algorithm.estimates.main1_generate_cost_table_data import read_buckets
from facto.algorithm.prep import table_str


def generate_cost_table(*, csv_path: str | pathlib.Path, score_func: Callable[[int, int], float], out_path: str | pathlib.Path | None = None):
    buckets = read_buckets(csv_path)

    k = 0
    entries = {}
    entries[-1 - 2j] = '$n$'
    entries[0 - 2j] = '$s$'
    entries[1 - 2j] = r'$\ell$'
    entries[2 - 2j] = '$w_1$'
    entries[3 - 2j] = '$w_3$'
    entries[4 - 2j] = '$w_4$'
    entries[5 - 2j] = '$f$'
    entries[6 - 2j] = r'$m$'
    entries[7 - 2j] = r'$P_{\text{deviant}}$'
    entries[8 - 2j] = r'E(shots)'
    entries[9 - 2j] = r'Toffolis'
    entries[10- 2j] = r'Qubits'
    for k2 in range(-1, 11):
        entries[k2 - 1j] = '---'
    for n, v in sorted(buckets.items()):
        tup = min(v.items(), key=lambda e: score_func(e[0], e[1]["tofs"]))
        result = tup[1]
        att = CostConfig.from_params(
            modulus_bitlength=result["modulus_bitlength"],
            num_input_qubits=result["num_input_qubits"],
            num_shots=result["num_shots"],
            pp_success_probability=result["pp_success_probability"],
            l=result["l"],
            w1=result["w1"],
            w3=result["w3"],
            w4=result["w4"],
            len_acc=result["len_acc"])

        details = result["details"]
        if "s" in details.keys():
            s = int(details["s"])
        else:
            # Not relevant/available for this parameterization.
            s = "--"

        tof_str = f'{att.toffolis:0.2g}'
        if '.' not in tof_str:
            tof_str = tof_str.replace('e', '.0e')
        entries[-1 + k*1j] = att.conf.modulus.bit_length()
        entries[0 + k*1j] = s
        entries[1 + k*1j] = att.conf.rns_primes_bit_length
        entries[2 + k*1j] = att.conf.window1
        entries[3 + k*1j] = att.conf.window3a
        entries[4 + k*1j] = att.conf.window4
        entries[5 + k*1j] = att.conf.len_accumulator
        entries[6 + k*1j] = att.conf.num_input_qubits
        entries[7 + k*1j] = f'{att.conf.probability_of_deviation_failure:0.2%}'.replace('%', '\\%')
        entries[8 + k*1j] = f'{att.conf.expected_shots:0.1f}'
        entries[9 + k*1j] = tof_str
        entries[10+ k*1j] = att.conf.estimated_logical_qubits

        k += 1
    if out_path is not None:
        f: Any
        with open(out_path, 'w') as f:
            print(table_str(
                entries,
                latex=True,
            ), file=f)
    else:
        print(table_str(entries))

    if out_path is not None:
        print(f'wrote file://{pathlib.Path(out_path).absolute()}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', required=True, type=str)
    parser.add_argument('--out_path', default=None, type=str)
    parser.add_argument('--toffoli_power', required=True, type=int)
    parser.add_argument('--qubit_power', required=True, type=int)
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()

    csv_path = pathlib.Path(args.csv_path)
    if args.out_path is None:
        out_path = None
    else:
        out_path = pathlib.Path(args.out_path)
        out_path.parent.mkdir(exist_ok=True, parents=True)
    tof_power = args.toffoli_power
    qub_power = args.qubit_power

    generate_cost_table(
        csv_path=csv_path,
        score_func=lambda qubits, toffolis: toffolis**tof_power * qubits**qub_power,
        out_path=out_path,
    )


if __name__ == "__main__":
    main()
