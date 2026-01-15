from __future__ import annotations

import argparse
import dataclasses
import functools
import math
import pathlib

from facto.algorithm.estimates._cost_config import CostConfig
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
        return 3 * 4 * self.factory_count

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
    def hot_patches(self) -> int:
        height = 3 * self.factory_count
        routing_width = 3
        width = routing_width + math.ceil(self.min_hot_storage / height)
        return height * width

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

    cost_conf = CostConfig.from_params(
        modulus_bitlength=2048,
        num_input_qubits=math.ceil(2048 * (1/2 + 1/9)),
        num_shots=9,
        pp_success_probability=0.99,
        l=20,
        w1=6,
        w3=3,
        w4=5,
        len_acc=33,
    )
    coster = AlgoCoster(
        conf=cost_conf,
        surgery_micros=25,
        reaction_micros=10,
        factory_micros=150,
        factory_count=6,
    )

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
        (2*(d + 1)**2, 3.5**-d / 40)
        for d in range(50)
    ]

    target_hot_err = 1 / (coster.hot_patches * coster.shot_duration_micros * 10)
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
    print(f"{target_hot_err=}")
    print(f"{target_cold_err=}")
    print(f"{hot_size=}")
    print(f"{cold_size=}")

    total_qubits = 0
    total_qubits += coster.hot_patches * hot_size
    total_qubits += coster.factory_patches * hot_size
    total_qubits += coster.cold_logical_qubits * cold_size
    print(f'{coster.hot_patches=}')
    print(f'{coster.shot_success_rate=}')
    print(f'{coster.factory_patches=}')
    print(coster.total_duration_micros / 10**6 / 60 / 60 / 24, "days")
    print(f'{total_qubits=}')


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
