from __future__ import annotations

import argparse
import pathlib
from typing import Any, Callable

from facto.algorithm.estimates._cost_config import CostConfig
from facto.algorithm.estimates.main1_generate_cost_table_data import read_buckets

def generate_cost_table(
        *,
        csv_path: str | pathlib.Path,
        score_func: Callable[[int, int], float],
        out_path: str | pathlib.Path | None = None,
):
    buckets = read_buckets(csv_path)

    table_data = ""
  
    for _, v in sorted(buckets.items()):
        tup = min(v.items(), key=lambda e: score_func(e[0], e[1]["tofs"]))
        result: dict[str, Any] = tup[1]

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

        # n, z
        table_data += str(result["modulus_bitlength"]) + " & "
        if "z" in details.keys():
            table_data += str(details["z"]) + " & "
        else:
            # Not relevant/available for this parameterization.
            table_data += "---" + "  &  "

        # m_A, s_A, l_A, [sigma_a], P_pp, n_A
        table_data += str(details["m"]) + " & "
        if "s" in details.keys():
            table_data += str(details["s"]) + " & "
        else:
            # Not relevant/available for this parameterization.
            table_data += "---" + " & "
        table_data += str(details["l"]) + " & "
        if "sigma" in details.keys():
          table_data += str(details["sigma"]) + " & "

        P_pp = result["pp_success_probability"]
        P_fail = 1 - P_pp
        table_data += f'{P_fail:.0e}' + "  &  "

        table_data += str(result["num_shots"]) + "  &  "

        # m, \ell, w_1, w_3, w_4, f, P_\text{deviate}
        table_data += str(result["num_input_qubits"]) + "  &  "

        table_data += str(result["l"]) + " & "
        table_data += str(result["w1"]) + " & "
        table_data += str(result["w3"]) + " & "
        table_data += str(result["w4"]) + " & "
        table_data += str(result["len_acc"]) + " & "

        P_dev = f'{att.conf.probability_of_deviation_failure:0.2%}'
        P_dev = P_dev.replace('%', '\\%')
        table_data += P_dev + "  &  "

        # E(shots)
        exp_shots = att.conf.expected_shots
        table_data += f'{exp_shots:0.1f}' + "  &  "

        # Toffolis (per shot), Toffolis (overall), Qubits
        tof_overall = att.toffolis
        tof_per_shot = att.toffolis / exp_shots

        tof_str = f'{tof_per_shot:0.2g}'
        if '.' not in tof_str:
            tof_str = tof_str.replace('e', '.0e')
        table_data += str(tof_str) + " & "

        tof_str = f'{tof_overall:0.2g}'
        if '.' not in tof_str:
            tof_str = tof_str.replace('e', '.0e')
        table_data += tof_str + " & "

        table_data += str(att.conf.estimated_logical_qubits) + " \\\\\n"

    if out_path is not None:
        f: Any
        with open(out_path, 'w') as f:
            print(table_data, file=f)
    else:
        print(table_data)

    if out_path is not None:
        print(f'wrote file://{pathlib.Path(out_path).absolute()}')



"""
    |
    | parameterizations/create-csv-files.py
    |     (turns hardcoded parameters from https://github.com/ekera/qunundrum into shot counts etc)
    v
CSV[modulus_bitlength,num_input_qubits,num_shots,pp_success_probability,details]
    |
    | src/factor/algorithm/estimates/main1_generate_cost_table_data.py
    |     (picks execution parameters like w1 by doing a grid scan)
    v
CSV[n,s,l,w1,w3,w4,len_acc,tofs,qubits]
    |
    | src/factor/algorithm/estimates/main5_generate_cost_table.py
    v
CostTableInPaper (using CostConfig which uses ExecutionConfig which uses ExecutionConfig which uses ProblemConfig)
"""


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
