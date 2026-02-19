"""Data flow into this file:

parameterizations/create-csv-files.py
    |
    |     (turns hardcoded parameters from https://github.com/ekera/qunundrum into shot counts etc)
    v
CSV[modulus_bitlength,num_input_qubits,num_shots,pp_success_probability,details]
    |
    | src/factor/algorithm/estimates/main1_generate_cost_table_data.py
    |     (picks execution parameters like w1 by doing a grid scan)
    v
CSV[n,s,l,w1,w3,w4,len_acc,tofs,qubits]
    |
    | src/factor/algorithm/estimates/main6_generate_physical_estimate.py (this file)
    |      (using CostConfig which uses ExecutionConfig which uses ExecutionConfig which uses ProblemConfig)
    v
LaTeX table used in paper
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import math
import pathlib
from typing import Any

from matplotlib import pyplot as plt

from facto.algorithm.estimates._cost_config import CostConfig, min_pareto_frontier
from facto.algorithm.estimates.main1_generate_cost_table_data import read_buckets
import sinter

from facto.algorithm.estimates._cost_config import CostConfig
from facto.algorithm.estimates.main1_generate_cost_table_data import read_buckets, read_csv_for_table_data
from facto.algorithm.prep import table_str
from facto.operations.adder._add_coster import AddCoster
from facto.operations.lookup._lookup_coster import LookupCoster
from facto.operations.phaseup._phaseup_coster import PhaseupCoster
from scatter_script import CostKey


@dataclasses.dataclass(frozen=True)
class AlgoCoster:
    conf: CostConfig
    surgery_micros: float
    reaction_micros: float
    factory_micros: float
    factory_count: int

    @functools.cached_property
    def factory_patches(self) -> int:
        w, h = self.factory_span
        return w * h

    @functools.cached_property
    def min_hot_storage(self) -> int:
        c = self.conf.conf
        hot_stored = (
            + c.num_input_qubits.bit_length()  # Loop 1 -> Loop 2 -> Loop 3 value (log of residue).
            + 2*c.rns_primes_bit_length  # Loop 3 -> Loop 4 value (residue).
            + 3*c.len_accumulator  # Loop 4 mutated value (accumulator).
        )
        return hot_stored

    @functools.cached_property
    def hot_storage_patches(self) -> int:
        w, h = self.hot_span
        return w * h

    @functools.cached_property
    def compute_area_patches(self) -> int:
        w1, h1 = self.routing_span
        w2, h2 = self.factory_span
        return w1*h1 + w2*h2

    @functools.cached_property
    def hot_span(self) -> tuple[int, int]:
        height = 3 * self.factory_count
        width = math.ceil(self.min_hot_storage / height)
        return width, height

    @functools.cached_property
    def routing_span(self) -> tuple[int, int]:
        return 3, 3 * self.factory_count

    @functools.cached_property
    def factory_span(self) -> tuple[int, int]:
        return 4, 3 * self.factory_count

    @functools.cached_property
    def cold_logical_qubits(self) -> int:
        return self.conf.conf.num_input_qubits

    @functools.cached_property
    def shot_success_rate(self) -> float:
        return self.conf.conf.pp_success_probability * (1 - self.conf.conf.probability_of_deviation_failure)

    @functools.cached_property
    def total_duration_micros(self) -> float:
        return self.shot_duration_micros * self.conf.conf.num_shots / self.shot_success_rate

    @functools.cached_property
    def shot_duration_micros(self) -> float:
        step_data: dict[str, dict[CostKey, int]] = self.conf.conf.estimated_subroutine_costs
        duration_micros = 0
        for step_name, step_costs in step_data.items():
            for op, reps in step_costs.items():
                # Slack for rearranging qubits to prepare for subroutine.
                duration_micros += self.surgery_micros * 10 * reps

                if op.name == 'init_lookup':
                    c = LookupCoster(
                        addr_len=(op['N'] - 1).bit_length(),
                        surgery_micros=self.surgery_micros,
                        reaction_micros=self.reaction_micros,
                        factory_micros=self.factory_micros,
                        factory_count=self.factory_count,
                    )
                    duration_micros += c.lookup_duration_micros * reps
                elif op.name == '__iadd__' or op.name == 'phase_flip_if_cmp':
                    c = AddCoster(
                        n=op['n'],
                        surgery_micros=self.surgery_micros,
                        reaction_micros=self.reaction_micros,
                        factory_micros=self.factory_micros,
                        factory_count=self.factory_count,
                    )
                    duration_micros += c.adder_duration_micros * reps
                elif op.name == 'phaseflip_by_lookup':
                    c = PhaseupCoster(
                        addr_len=(op['N'] - 1).bit_length(),
                        surgery_micros=self.surgery_micros,
                        reaction_micros=self.reaction_micros,
                        factory_micros=self.factory_micros,
                        factory_count=self.factory_count,
                    )
                    duration_micros += c.phaseup_duration_micros * reps
                else:
                    raise NotImplementedError(f'{op=}, {reps=}')
        return duration_micros


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

    data = read_csv_for_table_data(csv_path)
    grouped_data = sinter.group_by(data, key=lambda e: (e["modulus_bitlength"], e["num_input_qubits"]))
    score_func = lambda qubits, toffolis: toffolis ** tof_power * qubits ** qub_power
    entries = {}

    col = 0
    entries[-2j + col] = "Modulus"
    entries[-1j + col] = "Bits (n)"
    col += 1
    entries[-2j + col] = "Input"
    entries[-1j + col] = "Qubits (m)"
    col += 1
    entries[-2j + col] = "Hot"
    entries[-1j + col] = "Area"
    col += 1
    entries[-2j + col] = "Compute"
    entries[-1j + col] = "Area"
    col += 1
    # entries[-1j + col] = "target hot err"
    # col += 1
    # entries[-1j + col] = "target cold err"
    # col += 1
    entries[-2j + col] = "Physical Qubits"
    entries[-1j + col] = "in Compute Area"
    col += 1
    entries[-2j + col] = "Physical Qubits"
    entries[-1j + col] = "in Hot Area"
    col += 1
    entries[-2j + col] = "Physical Qubits"
    entries[-1j + col] = "in Cold Area"
    col += 1
    entries[-2j + col] = "CCZ"
    entries[-1j + col] = "Factories"
    col += 1
    entries[-1j + col] = "Runtime"
    entries[-2j + col] = "(days)"
    col += 1
    entries[-2j + col] = "physical"
    entries[-1j + col] = "qubits"
    col += 1
    entries[-2j + col] = "shots"
    entries[-1j + col] = ""
    col += 1
    entries[-2j + col] = "shot success"
    entries[-1j + col] = "rate"
    col += 1
    entries[-2j + col] = "pp success"
    entries[-1j + col] = "rate"
    col += 1
    entries[-1j + col] = "l"
    col += 1
    entries[-1j + col] = "w1"
    col += 1
    entries[-1j + col] = "w3"
    col += 1
    entries[-1j + col] = "w4"
    col += 1
    entries[-2j + col] = "len"
    entries[-1j + col] = "acc"

    # Eyeballed minimum storage costs from yoked storage cost plot.
    cold_confs = [
        (190, 1e-7),
        (240, 1e-9),
        (300, 1e-11),
        (360, 1e-13),
        (420, 1e-15),
        (500, 1e-17),
        (600, 1e-19),
    ]
    # Non-yoked storage cost (fit parameters taken from yoked storage cost plot).
    hot_confs = [
        (2 * (d + 1) ** 2, 3.5 ** -d / 40)
        for d in range(50)
    ]

    for row, (_group_key, group) in enumerate(sorted(grouped_data.items())):
        if True:
            xys = []
            for tup in group:
                result: dict[str, Any] = tup

                cost_conf = CostConfig.from_params(
                    modulus_bitlength=result["modulus_bitlength"],
                    num_input_qubits=result["num_input_qubits"],
                    num_shots=result["num_shots"],
                    pp_success_probability=result["pp_success_probability"],
                    l=result["l"],
                    w1=result["w1"],
                    w3=result["w3"],
                    w4=result["w4"],
                    len_acc=result["len_acc"],
                )

                coster = AlgoCoster(
                    conf=cost_conf,
                    surgery_micros=25,
                    reaction_micros=10,
                    factory_micros=150,
                    factory_count=6,
                )

                hot_patches = coster.hot_storage_patches + coster.compute_area_patches - coster.factory_patches
                target_hot_err = 1 / (hot_patches * coster.shot_duration_micros * 10)
                target_cold_err = 1 / (coster.cold_logical_qubits * coster.shot_duration_micros * 10)
                hot_size = min(
                    qubits
                    for qubits, err_per_round_per_logical in hot_confs
                    if err_per_round_per_logical <= target_hot_err
                )
                cold_size = min(
                    qubits
                    for qubits, err_per_round_per_logical in cold_confs + hot_confs
                    if err_per_round_per_logical <= target_cold_err
                )

                total_qubits = 0
                total_qubits += hot_patches * hot_size
                total_qubits += coster.factory_patches * hot_size
                total_qubits += coster.cold_logical_qubits * cold_size

                duration_days = coster.total_duration_micros / 10 ** 6 / 60 / 60 / 24
                xys.append((duration_days, total_qubits))
            xys = min_pareto_frontier(xys)
            plt.plot([e[0] for e in xys], [e[1] for e in xys], label=coster.conf.conf.modulus.bit_length(), marker='x')

        result: dict[str, Any] = min(group, key=lambda e: score_func(e["qubits"], e["tofs"]))

        cost_conf = CostConfig.from_params(
            modulus_bitlength=result["modulus_bitlength"],
            num_input_qubits=result["num_input_qubits"],
            num_shots=result["num_shots"],
            pp_success_probability=result["pp_success_probability"],
            l=result["l"],
            w1=result["w1"],
            w3=result["w3"],
            w4=result["w4"],
            len_acc=result["len_acc"],
        )

        coster = AlgoCoster(
            conf=cost_conf,
            surgery_micros=25,
            reaction_micros=10,
            factory_micros=150,
            factory_count=6,
        )

        hot_patches = coster.hot_storage_patches + coster.compute_area_patches - coster.factory_patches
        target_hot_err = 1 / (hot_patches * coster.shot_duration_micros * 10)
        target_cold_err = 1 / (coster.cold_logical_qubits * coster.shot_duration_micros * 10)
        hot_size = min(
            qubits
            for qubits, err_per_round_per_logical in hot_confs
            if err_per_round_per_logical <= target_hot_err
        )
        cold_size = min(
            qubits
            for qubits, err_per_round_per_logical in cold_confs + hot_confs
            if err_per_round_per_logical <= target_cold_err
        )

        total_qubits = 0
        total_qubits += hot_patches * hot_size
        total_qubits += coster.factory_patches * hot_size
        total_qubits += coster.cold_logical_qubits * cold_size

        duration_days = coster.total_duration_micros / 10**6 / 60 / 60 / 24

        col = 0
        entries[1j*row + col] = result["modulus_bitlength"]
        col += 1
        entries[1j*row + col] = result["num_input_qubits"]
        col += 1
        entries[1j*row + col] = f"{coster.hot_span[0]} x {coster.hot_span[1]}"
        col += 1
        entries[1j*row + col] = f"{coster.routing_span[0] + coster.factory_span[0]} x {coster.routing_span[1]}"
        col += 1
        # entries[1j*row + col] = target_hot_err
        # col += 1
        # entries[1j*row + col] = target_cold_err
        # col += 1
        entries[1j*row + col] = hot_size * coster.compute_area_patches
        col += 1
        entries[1j*row + col] = hot_size * coster.hot_storage_patches
        col += 1
        entries[1j*row + col] = cold_size * coster.cold_logical_qubits
        col += 1
        entries[1j*row + col] = coster.factory_count
        col += 1
        entries[1j*row + col] = duration_days
        col += 1
        entries[1j*row + col] = total_qubits
        col += 1
        entries[1j*row + col] = result["num_shots"]
        col += 1
        entries[1j*row + col] = coster.shot_success_rate
        col += 1
        entries[1j*row + col] = result["pp_success_probability"]
        col += 1
        entries[1j*row + col] = result["l"]
        col += 1
        entries[1j*row + col] = result["w1"]
        col += 1
        entries[1j*row + col] = result["w3"]
        col += 1
        entries[1j*row + col] = result["w4"]
        col += 1
        entries[1j*row + col] = result["len_acc"]
    def formatter(e: float | int) -> str:
        if isinstance(e, float):
            return f'{e:0.2f}'
        return str(e)

    s = table_str(entries, latex=True, formatter=formatter)
    if out_path is None:
        print(s)
    else:
        with open(out_path, "w") as f:
            print(s, file=f)

    plt.ylabel("Physical Qubits")
    plt.xlabel("Days")
    plt.loglog()
    plt.legend()
    plt.show()


def hot_error(rounds: int, block_size: int, distance: int) -> float:
    # These fit parameters are listed in the figure titled
    #   Projections of the cost of storage of normal surface codes and surface codes
    #   “yoked by” (concatenated below) 1D and 2D parity check codes, under uniform depolarizing noise.
    return rounds * block_size * 3.5**-distance / 40


def cold_error(rounds: int, block_size: int, distance: int) -> float:
    # These fit parameters are listed in the figure titled
    #   Projections of the cost of storage of normal surface codes and surface codes
    #   “yoked by” (concatenated below) 1D and 2D parity check codes, under uniform depolarizing noise.
    return rounds**4 * block_size**2 * 150**-distance / 50_000


if __name__ == "__main__":
    main()
