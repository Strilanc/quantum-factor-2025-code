from __future__ import annotations

import argparse
import csv
import itertools
import json
import multiprocessing
import pathlib
import sys
from typing import Any

from facto.algorithm.estimates._cost_config import CostConfig


def generate_cost_table_data(
        *,
        params_csv_path: str | pathlib.Path,
        out_csv_path: str | pathlib.Path,
):
    print(f"Loading the parameterizations...", file=sys.stderr)
    parameterizations = read_parameterizations(params_csv_path)

    print(f"Generating the grid scan...", file=sys.stderr)
    tups = itertools.product(
        parameterizations,
        range(24, 60),  # len_acc
        range(18, 26),  # l
    )

    pool = multiprocessing.Pool()

    f: Any
    with open(out_csv_path, 'w') as f:
        csv_writer = csv.DictWriter(f, fieldnames=[
            "modulus_bitlength",
            "num_input_qubits",
            "num_shots",
            "pp_success_probability",
            "l",
            "w1",
            "w3",
            "w4",
            "len_acc",
            "tofs",
            "qubits",
            "details"])
        csv_writer.writeheader()

        for results in pool.imap(CostConfig.iter_configurations, tups):
            csv_writer.writerows(results)
  
    print(f"wrote file://{pathlib.Path(out_csv_path).absolute()}", file=sys.stderr)

def read_buckets(csv_path: str | pathlib.Path) -> dict[int, dict[int, tuple[int, ...]]]:
    buckets: dict[int, dict[int, tuple[int, ...]]] = {}
    f: Any
    with open(csv_path) as f:
        csv_reader = csv.DictReader(f)
        for result in csv_reader:
            n = result["modulus_bitlength"] = int(result["modulus_bitlength"])
            result["num_input_qubits"] = int(result["num_input_qubits"])
            result["num_shots"] = int(result["num_shots"])
            result["pp_success_probability"] = float(result["pp_success_probability"])

            result["l"] = int(result["l"])
            result["w1"] = int(result["w1"])
            result["w3"] = int(result["w3"])
            result["w4"] = int(result["w4"])
            result["len_acc"] = int(result["len_acc"])
            tofs = result["tofs"] = int(result["tofs"])
            qubits = result["qubits"] = int(result["qubits"])

            result["details"] = json.loads(result["details"])

            if n not in buckets:
                buckets[n] = {}
            if qubits not in buckets[n] or tofs < buckets[n][qubits]["tofs"]:
                buckets[n][qubits] = result

    return buckets

def read_parameterizations(params_csv_path: str | pathlib.Path):
    parameterizations = []

    with open(params_csv_path) as f:
        csv_reader = csv.DictReader(f)
        for params in csv_reader:
            params["modulus_bitlength"] = int(params["modulus_bitlength"])
            params["num_input_qubits"] = int(params["num_input_qubits"])
            params["num_shots"] = int(params["num_shots"])
            params["pp_success_probability"] = float(params["pp_success_probability"])
            params["details"] = json.loads(params["details"])
            parameterizations.append(params)

    return parameterizations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--params_csv_path', required=True, type=str)
    parser.add_argument('--out_csv_path', required=True, type=str)
    args = parser.parse_args()

    params_csv_path = pathlib.Path(args.params_csv_path)

    out_csv_path = pathlib.Path(args.out_csv_path)
    out_csv_path.parent.mkdir(exist_ok=True, parents=True)

    generate_cost_table_data(
        params_csv_path=params_csv_path,
        out_csv_path=out_csv_path
    )


if __name__ == "__main__":
    main()
