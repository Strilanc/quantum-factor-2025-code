from __future__ import annotations

import collections
import dataclasses
import functools
import math
import pathlib
import sys
import time
from typing import Any, Mapping

import numpy as np

from facto.algorithm.prep._precompute_multipliers import find_multipliers_for_conf
from facto.algorithm.prep._precompute_rns import find_rns_for_conf
from facto.algorithm.prep._precompute_generators import precompute_generators
from facto.algorithm.prep._precompute_table1 import precompute_table1
from facto.algorithm.prep._precompute_table3 import precompute_table3
from facto.algorithm.prep._precompute_table4 import precompute_table4
from facto.algorithm.prep._problem_config import ProblemConfig
from scatter_script import Lookup, CostKey, ZeroArray


@dataclasses.dataclass
class ExecutionConfig:
    """Runtime configuration information for executing quantum factoring.

    Args:
        conf: The problem configuration (specifies basic parameters used to generate the
            full execution config).
        periods: The prime numbers used by the residue number system.
        generators: Multiplicative generators for each prime number used by the residue number
            system.
        table1: Offset constants for the discrete-log-of-residue accumulation loop.
        table3a: Offset constants for the discrete-log-to-residue loop.
        table3b: Offset constants for the discrete-log-to-residue loop.
        table3c: Initialization lookups for the discrete-log-to-residue loop.
        table4: Offset constants for the truncated-residue-accumulation loop.
    """

    conf: ProblemConfig
    periods: np.ndarray
    generators: np.ndarray
    table1: np.ndarray | ZeroArray
    table3a: np.ndarray | ZeroArray
    table3b: np.ndarray | ZeroArray
    table3c: np.ndarray | ZeroArray
    table4: np.ndarray | ZeroArray

    def with_edits(self, conf: ProblemConfig | None) -> ExecutionConfig:
        return ExecutionConfig(
            conf=self.conf if conf is None else conf,
            periods=self.periods,
            generators=self.generators,
            table1=self.table1,
            table3a=self.table3a,
            table3b=self.table3b,
            table3c=self.table3c,
            table4=self.table4,
        )

    @property
    def lookup1(self) -> Lookup:
        return Lookup(self.table1)

    @property
    def lookup3a(self) -> Lookup:
        return Lookup(self.table3a)

    @property
    def lookup3b(self) -> Lookup:
        return Lookup(self.table3b)

    @property
    def lookup3c(self) -> Lookup:
        return Lookup(self.table3c)

    @property
    def lookup4(self) -> Lookup:
        return Lookup(self.table4)

    @property
    def estimated_ideal_mask_bits(self) -> int:
        accumulator_additions = self.num_windows4 * len(self.periods)
        deviation_per_addition = 3 * 2**-self.len_accumulator
        eps = accumulator_additions * deviation_per_addition
        return round(math.log2(eps) / 2 + self.len_accumulator)

    @staticmethod
    def vacuous_config_for_problem(
        problem_conf: ProblemConfig, *, override_num_periods: int | None = None
    ) -> ExecutionConfig:
        if problem_conf.rns_primes_bit_length is None:
            problem_conf = problem_conf.with_edits(
                rns_primes_bit_length=problem_conf.estimate_minimum_rns_period_bit_length()
            )
        if problem_conf.rns_primes_range_start is None:
            problem_conf = problem_conf.with_edits(rns_primes_range_start=-1)
        if problem_conf.rns_primes_range_stop is None:
            problem_conf = problem_conf.with_edits(rns_primes_range_stop=-1)
        num_periods = override_num_periods
        if num_periods is None:
            num_periods = math.ceil(
                problem_conf.modulus.bit_length()
                * problem_conf.num_input_qubits
                / problem_conf.window1
                / problem_conf.rns_primes_bit_length
            )

        shape1 = (num_periods + 1, problem_conf.num_windows1, 1 << problem_conf.window1)
        shape3ab = (
            num_periods,
            problem_conf.num_windows3a,
            problem_conf.num_windows3b,
            1 << (problem_conf.window3a + problem_conf.window3b),
        )
        shape3c = (num_periods, 1 << (2 * problem_conf.window3a))
        shape4 = (num_periods, problem_conf.num_windows4, 1 << problem_conf.window4)
        return ExecutionConfig(
            conf=problem_conf,
            periods=np.zeros(num_periods, dtype=np.uint32)
            + (1 << (problem_conf.rns_primes_bit_length - 1)),
            generators=np.ones(num_periods, dtype=np.uint32),
            table1=ZeroArray(shape=shape1, dtype=np.uint64),
            table3a=ZeroArray(shape=shape3ab, dtype=np.uint64),
            table3b=ZeroArray(shape=shape3ab, dtype=np.uint64),
            table3c=ZeroArray(shape=shape3c, dtype=np.uint64),
            table4=ZeroArray(shape=shape4, dtype=np.uint64),
        )

    @staticmethod
    def vacuous_config(
        *,
        modulus_bitlength: int,
        num_input_qubits: int,
        num_shots: int,
        pp_success_probability: float,
        num_periods: int,
        period_bitlength: int,
        w1: int = 1,
        w4: int = 1,
        w3a: int = 1,
        w3b: int = 1,
        len_acc: int = 24,
        mask_bits: int | None = None,
    ) -> ExecutionConfig:
        problem_conf = ProblemConfig(
            modulus=(1 << modulus_bitlength) - 1,
            num_input_qubits=num_input_qubits,
            pp_success_probability=pp_success_probability,
            num_shots=num_shots,
            generator=2,
            window1=w1,
            window3a=w3a,
            window3b=w3b,
            window4=w4,
            min_wraparound_gap=len_acc,
            len_accumulator=len_acc,
            parallelism=1,
            rns_primes_bit_length=period_bitlength,
            rns_primes_range_start=2 ** (period_bitlength - 1),
            rns_primes_range_stop=2**period_bitlength,
            rns_primes_skipped=(),
            rns_primes_extra=(),
            mask_bits=len_acc >> 1 if mask_bits is None else mask_bits,
        )
        result = ExecutionConfig.vacuous_config_for_problem(
            problem_conf, override_num_periods=num_periods
        )
        if mask_bits is None:
            result = result.with_edits(
                conf=problem_conf.with_edits(mask_bits=result.estimated_ideal_mask_bits)
            )
        return result

    def __post_init__(self):
        if self.conf.rns_primes_range_start is None:
            raise ValueError(
                f"Execution config requires its ProblemConfig to be solved, but {self.conf.rns_primes_range_start=}."
            )

    @property
    def probability_of_deviation_failure(self) -> float:
        accumulator_additions = self.num_windows4 * len(self.periods)
        deviation_per_addition = 2 * 2**-self.len_accumulator + 2**-self.min_wraparound_gap
        deviation = accumulator_additions * deviation_per_addition
        mask_proportion = 2 ** (self.mask_bits - self.len_accumulator)
        infidelity_from_revelation = deviation / mask_proportion
        err = mask_proportion + infidelity_from_revelation
        return err

    @property
    def mask_bits(self) -> int:
        return self.conf.mask_bits

    @property
    def modulus(self) -> int:
        return self.conf.modulus

    @property
    def num_input_qubits(self) -> int:
        return self.conf.num_input_qubits

    @property
    def generator(self) -> int:
        return self.conf.generator

    @property
    def window1(self) -> int:
        return self.conf.window1

    @property
    def window3a(self) -> int:
        return self.conf.window3a

    @property
    def window3b(self) -> int:
        return self.conf.window3b

    @property
    def window4(self) -> int:
        return self.conf.window4

    @property
    def min_wraparound_gap(self) -> int:
        return self.conf.min_wraparound_gap

    @property
    def len_accumulator(self) -> int:
        return self.conf.len_accumulator

    @property
    def parallelism(self) -> int:
        return self.conf.parallelism

    @property
    def num_shots(self) -> int:
        return self.conf.num_shots

    @property
    def pp_success_probability(self) -> int:
        return self.conf.pp_success_probability

    @property
    def expected_shots(self) -> float:
        return self.conf.num_shots / (1 - self.probability_of_deviation_failure) / self.conf.pp_success_probability

    @property
    def estimated_logical_qubits(self) -> int:
        return (
            self.num_input_qubits  # Cold storage.
            + self.len_dlog_accumulator  # Loop 1 -> Loop 2 -> Loop 3 value (log of residue).
            + self.rns_primes_bit_length  # Loop 3 -> Loop 4 value (residue).
            + self.len_accumulator  # Loop 4 mutated value (accumulator).
            + max(
                self.rns_primes_bit_length, self.len_accumulator, self.len_dlog_accumulator
            )  # Lookup register
        )

    @property
    def rns_primes_bit_length(self) -> int | None:
        return self.conf.rns_primes_bit_length

    @property
    def rns_primes_range_start(self) -> int | None:
        return self.conf.rns_primes_range_start

    @property
    def rns_primes_range_stop(self) -> int | None:
        return self.conf.rns_primes_range_stop

    @property
    def rns_primes_extra(self) -> tuple[int, ...]:
        return self.conf.rns_primes_extra

    @property
    def rns_primes_skipped(self) -> tuple[int, ...]:
        return self.conf.rns_primes_skipped

    @property
    def len_dlog_accumulator(self) -> int:
        return self.conf.len_dlog_accumulator

    @property
    def truncated_modulus(self) -> int:
        return self.conf.truncated_modulus

    @property
    def num_windows1(self) -> int:
        return self.conf.num_windows1

    @property
    def num_windows3a(self) -> int:
        return self.conf.num_windows3a

    @property
    def num_windows3b(self) -> int:
        return self.conf.num_windows3b

    @property
    def num_windows4(self) -> int:
        return self.conf.num_windows4

    @property
    def dropped_bits(self) -> int:
        return self.conf.dropped_bits

    @functools.cached_property
    def estimated_subroutine_costs_per_call(self) -> dict[str, dict[CostKey, int]]:
        result = {}
        result["loop1"] = dict(self._estimate_cost_loop1())
        result["loop2"] = dict(self._estimate_cost_loop2())
        result["loop3"] = dict(self._estimate_cost_loop3())
        result["loop4"] = dict(self._estimate_cost_loop4())
        result["unloop3"] = dict(self._estimate_cost_unloop3())
        result["unloop2"] = dict(self._estimate_cost_unloop2())
        return result

    @functools.cached_property
    def estimated_subroutine_costs(self) -> dict[str, dict[CostKey, int]]:
        def scale_counter(c: Mapping[CostKey, int], s: int) -> dict[CostKey, int]:
            return {k: round(v * s) for k, v in c.items()}

        result = {}
        result["loop1"] = scale_counter(self._estimate_cost_loop1(), len(self.periods) + 1)
        result["loop2"] = scale_counter(self._estimate_cost_loop2(), len(self.periods))
        result["loop3"] = scale_counter(self._estimate_cost_loop3(), len(self.periods))
        result["loop4"] = scale_counter(self._estimate_cost_loop4(), len(self.periods))
        result["unloop3"] = scale_counter(self._estimate_cost_unloop3(), len(self.periods))
        result["unloop2"] = scale_counter(self._estimate_cost_unloop2(), len(self.periods))
        return result

    def _estimate_cost_loop1(self) -> collections.Counter[CostKey]:
        result = collections.Counter(
            {
                CostKey(
                    "init_lookup", {"N": 1 << self.window1, "w": self.len_dlog_accumulator}
                ): self.num_windows1,
                CostKey("__iadd__", {"n": self.len_dlog_accumulator}): self.num_windows1,
            }
        )
        return result

    def _estimate_cost_loop2(self) -> collections.Counter[CostKey]:
        result = collections.Counter()
        for k in range(self.rns_primes_bit_length, self.len_dlog_accumulator):
            result[CostKey("__iadd__", {"n": k})] += 1
            result[CostKey("__iadd__", {"n": k + 1})] += 1
            result[CostKey("init_lookup", {"N": 2, "w": k})] += 1
        return result

    def _estimate_cost_unloop2(self) -> collections.Counter[CostKey]:
        result = collections.Counter()
        for k in range(self.rns_primes_bit_length, self.len_dlog_accumulator):
            result[CostKey("__iadd__", {"n": k})] += 1
            result[CostKey("__iadd__", {"n": k + 1})] += 1
            result[CostKey("init_lookup", {"N": 2, "w": k})] += 1
        return result

    def _estimate_cost_loop3(self) -> collections.Counter[CostKey]:
        result = collections.Counter()

        inner_iters = max(0, self.num_windows3a - 2) * self.num_windows3b
        result += collections.Counter(
            {
                CostKey(
                    "init_lookup",
                    {"N": 1 << (self.window3a * 2), "w": self.rns_primes_bit_length + 1},
                ): 1
            }
        )
        result += collections.Counter(
            {
                CostKey("__iadd__", {"n": self.rns_primes_bit_length + 1}): inner_iters,
                CostKey("__iadd__", {"n": self.rns_primes_bit_length}): inner_iters,
                CostKey(
                    "init_lookup",
                    {
                        "N": 1 << (self.window3a + self.window3b),
                        "w": self.rns_primes_bit_length + 1,
                    },
                ): inner_iters,
                CostKey("init_lookup", {"N": 2, "w": self.rns_primes_bit_length}): inner_iters,
            }
        )

        return result

    def _estimate_cost_loop4(self) -> collections.Counter[CostKey]:
        period_bit_length = self.rns_primes_bit_length
        w4 = self.window4
        len_acc = self.len_accumulator
        nw4 = -(-period_bit_length // w4)

        # Subtract the offset, with +1 for tracking underflow to get a comparison.
        result = collections.Counter(
            {
                CostKey("init_lookup", {"N": 1 << w4, "w": len_acc + 1}): nw4,
                CostKey("__iadd__", {"n": len_acc + 1}): nw4,
            }
        )

        # Expand underflow into modulus correction offset, then add it.
        result += collections.Counter(
            {
                CostKey("init_lookup", {"N": 2, "w": len_acc}): nw4,
                CostKey("__iadd__", {"n": len_acc}): nw4,
            }
        )

        # Uncompute the underflow (50% chance to not need the phasing).
        result += collections.Counter(
            {
                CostKey("init_lookup", {"N": 1 << w4, "w": len_acc}): nw4 * 0.5,
                CostKey("phase_flip_if_cmp", {"n": len_acc}): nw4 * 0.5,
            }
        )

        # Correct accumulated table uncomputations.
        result += collections.Counter({CostKey("phaseflip_by_lookup", {"N": 1 << w4}): nw4})

        return result

    def _estimate_cost_unloop3(self) -> collections.Counter[CostKey]:
        result = collections.Counter()

        inner_iters = max(0, self.num_windows3a - 2) * self.num_windows3b
        result += collections.Counter(
            {CostKey("phaseflip_by_lookup", {"N": 1 << (self.window3a * 2)}): 1}
        )
        result += collections.Counter(
            {
                CostKey("__iadd__", {"n": self.rns_primes_bit_length + 1}): inner_iters * 2,
                CostKey("__iadd__", {"n": self.rns_primes_bit_length}): inner_iters * 2,
                CostKey(
                    "init_lookup",
                    {
                        "N": 1 << (self.window3a + self.window3b),
                        "w": self.rns_primes_bit_length + 1,
                    },
                ): inner_iters
                * 2,
                CostKey("init_lookup", {"N": 2, "w": self.rns_primes_bit_length}): inner_iters * 2,
            }
        )
        result += collections.Counter(
            {
                CostKey("phase_flip_if_cmp", {"n": self.rns_primes_bit_length}): inner_iters,
                CostKey(
                    "init_lookup",
                    {"N": 1 << (self.window3a + self.window3b), "w": self.rns_primes_bit_length},
                ): inner_iters,
                CostKey(
                    "phaseflip_by_lookup", {"N": 1 << (self.window3a + self.window3b)}
                ): inner_iters
                * 2,
            }
        )

        return result

    @staticmethod
    def from_problem_config(
        conf: ProblemConfig, *, print_progress: bool = False
    ) -> ExecutionConfig:
        times = [time.monotonic()]
        if print_progress:
            print("Generating multipliers used by the windowed arithmetic...", file=sys.stderr)

        multipliers = find_multipliers_for_conf(conf)

        if print_progress:
            times.append(time.monotonic())
            dt = times[-1] - times[-2]
            dt = math.ceil(dt * 10) / 10
            print(f"    (took {dt}s)", file=sys.stderr)
            print("Finding prime periods for the residue system...", file=sys.stderr)

        rns_solution = find_rns_for_conf(
            conf, not_nil_constraints=multipliers.values(), print_progress=print_progress
        )
        conf = conf.with_edits(
            rns_primes_bit_length=rns_solution.primes_bit_length,
            rns_primes_range_start=rns_solution.primes_range_start,
            rns_primes_range_stop=rns_solution.primes_range_stop,
            rns_primes_extra=rns_solution.primes_extra,
            rns_primes_skipped=rns_solution.primes_skipped,
        )
        period_dtype: Any = np.uint64
        if conf.rns_primes_bit_length <= 32:
            period_dtype = np.uint32

        result_periods = np.array(rns_solution.periods, dtype=period_dtype)
        if print_progress:
            times.append(time.monotonic())
            dt = times[-1] - times[-2]
            dt = math.ceil(dt * 10) / 10
            print(f"    (took {dt}s)", file=sys.stderr)
            print("Finding multiplicative generators for each prime period...", file=sys.stderr)
        generators = precompute_generators(
            periods=rns_solution.periods, print_progress=print_progress
        )
        results_generators = np.array(generators, dtype=period_dtype)

        if print_progress:
            times.append(time.monotonic())
            print(f"    (took {math.ceil(times[-1] - times[-2])}s)", file=sys.stderr)
            print("Precomputing table 1 (windowed discrete log offsets)...", file=sys.stderr)
        results_table1_dlogs = precompute_table1(
            periods=rns_solution.periods,
            generators=generators,
            values=multipliers.values(),
            print_progress=print_progress,
            period_dtype=period_dtype,
        ).reshape((len(rns_solution.periods) + 1, conf.num_windows1, 1 << conf.window1))

        assert results_table1_dlogs.shape == (
            len(rns_solution.periods) + 1,
            conf.num_windows1,
            1 << conf.window1,
        )
        if print_progress:
            times.append(time.monotonic())
            dt = times[-1] - times[-2]
            dt = math.ceil(dt * 10) / 10
            print(f"    (took {dt}s)", file=sys.stderr)
            print("Precomputing table 3 (windowed modexp offsets)...", file=sys.stderr)
        table3a, table3b, table3c = precompute_table3(
            conf=conf,
            periods=rns_solution.periods,
            generators=generators,
            period_dtype=period_dtype,
            print_progress=print_progress,
        )
        if print_progress:
            times.append(time.monotonic())
            dt = times[-1] - times[-2]
            dt = math.ceil(dt * 10) / 10
            print(f"    (took {dt}s)", file=sys.stderr)
            print(
                "Precomputing table 4 (truncated residue accumulation offsets)...", file=sys.stderr
            )
        table4 = precompute_table4(
            conf=conf, periods=rns_solution.periods, print_progress=print_progress
        )
        if print_progress:
            times.append(time.monotonic())
            dt = times[-1] - times[-2]
            dt = math.ceil(dt * 10) / 10
            print(f"    (took {dt}s)", file=sys.stderr)
        return ExecutionConfig(
            conf=conf,
            periods=result_periods,
            generators=results_generators,
            table1=results_table1_dlogs,
            table3a=table3a,
            table3b=table3b,
            table3c=table3c,
            table4=table4,
        )

    @staticmethod
    def from_data_directory(root: pathlib.Path | str) -> ExecutionConfig:
        root = pathlib.Path(root)
        conf = ProblemConfig.from_ini_path(root / "problem_config.ini")
        period_t = np.uint32 if conf.rns_primes_bit_length <= 32 else np.uint64

        periods = np.fromfile(file=root / "periods.dat", dtype=period_t)
        generators = np.fromfile(file=root / "generators.dat", dtype=period_t)

        shape1 = (len(periods) + 1, conf.num_windows1, 1 << conf.window1)
        table1 = np.fromfile(root / "table1.dat", count=math.prod(shape1), dtype=period_t)
        table1 = table1.reshape(shape1)

        shape3ab = (
            len(periods),
            conf.num_windows3a,
            conf.num_windows3b,
            1 << (conf.window3a + conf.window3b),
        )
        shape3c = (len(periods), 1 << (conf.window3a * 2))
        table3a = np.fromfile(root / "table3a.dat", count=math.prod(shape3ab), dtype=period_t)
        table3b = np.fromfile(root / "table3b.dat", count=math.prod(shape3ab), dtype=period_t)
        table3c = np.fromfile(root / "table3c.dat", count=math.prod(shape3c), dtype=period_t)
        table3a = table3a.reshape(shape3ab)
        table3b = table3b.reshape(shape3ab)
        table3c = table3c.reshape(shape3c)

        shape4 = (len(periods), conf.num_windows4, 1 << conf.window4)
        table4 = np.fromfile(root / "table4.dat", count=math.prod(shape4), dtype=np.uint64)
        table4 = table4.reshape(shape4)

        return ExecutionConfig(
            periods=periods,
            conf=conf,
            generators=generators,
            table1=table1,
            table3a=table3a,
            table3b=table3b,
            table3c=table3c,
            table4=table4,
        )

    def write_to_data_directory(
        self,
        directory: pathlib.Path | str,
        *,
        print_progress: bool = False,
        input_conf: ProblemConfig | None | str = None,
    ):
        directory = pathlib.Path(directory)

        class _Announced:
            def __init__(self, name: str):
                self.path = directory / name

            def __enter__(self):
                return self.path

            def __exit__(self, exc_type, exc_val, exc_tb):
                if print_progress:
                    print(f"    wrote file://{self.path.absolute()}")

        with _Announced("README.txt") as p:
            with open(p, "w") as f:
                print(_READ_ME_CONTENTS.replace("{{MODULUS}}", str(self.conf.modulus)), file=f)

        if input_conf is not None:
            with _Announced("input.ini") as p:
                with open(p, "w") as f:
                    print(input_conf, file=f)

        with _Announced("problem_config.ini") as p:
            with open(p, "w") as f:
                print(self.conf, file=f)

        with _Announced("metadata.ini") as p:
            period_t = '"uint32_t"' if self.rns_primes_bit_length <= 32 else '"uint64_t"'
            with open(p, "w") as f:
                print(f"modulus_bit_length={self.conf.modulus.bit_length()}", file=f)
                print(f"len_accumulator={self.conf.len_accumulator}", file=f)
                print(f"dropped_bits={self.conf.dropped_bits}", file=f)
                print(f"period_t={period_t}", file=f)
                print(file=f)
                print(f"window1={self.conf.window1}", file=f)
                print(f"window3a={self.conf.window3a}", file=f)
                print(f"window3b={self.conf.window3b}", file=f)
                print(f"window4={self.conf.window4}", file=f)
                print(f"num_windows1={self.conf.num_windows1}", file=f)
                print(f"num_windows3a={self.conf.num_windows3a}", file=f)
                print(f"num_windows3b={self.conf.num_windows3b}", file=f)
                print(f"num_windows4={self.conf.num_windows4}", file=f)
                print(file=f)
                print(f"rns_primes_range_start={self.conf.rns_primes_range_start}", file=f)
                print(f"rns_primes_range_stop={self.conf.rns_primes_range_stop}", file=f)
                print(f"rns_primes_skipped={list(self.conf.rns_primes_skipped)}", file=f)
                print(f"rns_primes_extra={list(self.conf.rns_primes_extra)}", file=f)
                print(f"rns_num_primes={len(self.periods)}", file=f)
                print(
                    f"rns_total_bit_capacity={math.prod(int(e) for e in self.periods).bit_length() - 1}",
                    file=f,
                )
                print(file=f)
                print(f"len_dlog_accumulator={self.conf.len_dlog_accumulator}", file=f)
                print(file=f)
                tot = collections.Counter()
                for k, v in self.estimated_subroutine_costs.items():
                    for k2, v2 in simplified_costs(v).items():
                        tot[k2.name] += v2
                        print(f"estimated_shot_{k}_{k2.name}s={v2}", file=f)
                print(file=f)
                for k2, v2 in tot.items():
                    print(f"estimated_shot_{k2}s={v2}", file=f)
                print(file=f)
                tot = collections.Counter()
                for k, v in self.estimated_subroutine_costs.items():
                    for k2, v2 in black_box_costs(v, reaction_us=10, clifford_us=25).items():
                        tot[k2] += v2
                        print(f"estimated_shot_{k}_{k2}={v2}", file=f)
                print(file=f)
                for k2, v2 in tot.items():
                    print(f"estimated_shot_{k2}={v2}", file=f)
                print(file=f)
                for k2, v2 in tot.items():
                    print(f"estimated_factoring_{k2}={round(v2 * self.num_shots)}", file=f)

        with _Announced("periods.dat") as p:
            self.periods.tofile(p)
        with _Announced("generators.dat") as p:
            self.generators.tofile(p)
        with _Announced("table1.dat") as p:
            self.table1.tofile(p)
        with _Announced("table3a.dat") as p:
            self.table3a.tofile(p)
        with _Announced("table3b.dat") as p:
            self.table3b.tofile(p)
        with _Announced("table3c.dat") as p:
            self.table3c.tofile(p)
        with _Announced("table4.dat") as p:
            self.table4.tofile(p)


_READ_ME_CONTENTS = """This directory contains precomputed values for factoring N={{MODULUS}}.

The following files are expected to be present:

    - README.txt
    - input.ini
    - problem_config.ini
    - metadata.ini
    - periods.dat
    - generators.dat
    - table1.dat
    - table3a.dat
    - table3b.dat
    - table3c.dat
    - table4.dat

Here are more details about what each file contains.

=== README.txt ===
    Describes the meanings of the files, including itself.


=== input.ini ===
    A copy of the problem configuration file, before any unspecified values were autosolved.


=== problem_config.ini ===
    The problem configuration, with no unspecified values (all values filled in).


=== metadata.ini ===
    Various bits of summary information about the problem and configurations (like cost estimates).


=== periods.dat ===
    Binary data of the primes used by the residue number system.

    The file format is just a contiguous array of 32 or 64 bit unsigned integers, in the native
    endianness of the system that generated the file. Here is the format of the file, written as a
    C struct:
    
    struct PeriodsDataFormat {
        period_t periods[rns_num_primes];
    }

    Note that the size variables are listed in `metadata.ini`.
    The size of the file can be crosschecked against `rns_num_primes` listed in metadata.ini.


=== generators.dat ===
    Binary data of multiplicative generators for periods in the residue number system.

    The generators match up 1:1 with the periods from periods.dat. The file format is just a
    contiguous array of 32 or 64 bit unsigned integers, in the native endianness of the system that
    generated the file. Here is the format of the file, written as a C struct:

    struct PeriodsDataFormat {
        period_t generators[num_periods];
    }

    Note that the size variables are listed in `metadata.ini`.
    The size of the file can be crosschecked against `rns_num_primes` listed in metadata.ini.


=== table1.dat ===
    A table of precomputed offsets used for computing the discrete logarithms of residues.

    The table values satisfy:

        dlog_table[i, j, k] == dlog(generators[i], M[j, k], periods[i])
        table1 = np.diff(dlog_table, axis=0, prepend=[0], append=[0])

        where N is the number being factored
        where M[j, k] = pow(G, k << (j * window1), N) is a windowed multiplicand.
        where G is the randomly chosen classical base of the modexp operation in Shor's algorithm

    The format of the file is straight binary data: a series of 32 or 64 bit unsigned integers
    in the native byte endian-ness of the system that generated it. Here is the format of the file,
    written as a C struct:

        #if rns_primes_bit_length <= 32
            #define period_t uint32_t
        #else
            #define period_t uint64_t
        #endif
        struct Table1DataFormat {
            period_t dlogs[rns_num_periods + 1][num_windows1_dlogs][1 << window1];
        }

    Here is how to read the file using numpy:

        >>> period_t = np.uint32 if conf.rns_primes_bit_length <= 32 else np.uint64
        >>> shape1 = (rns_num_periods + 1, num_windows1_dlogs, 1 << window1)
        >>> table1 = np.fromfile('table1.dat', count=math.prod(shape1), dtype=period_t)
        >>> table1 = table1.reshape(shape1)

    Note that the size variables are listed in `metadata.ini`.


=== table3a.dat, table3b.dat, and table3c.dat ===
    Precomputed offsets used for the windowed arithmetic converting discrete logs into residues.

    The table values satisfy:

        table3a[i, ja, jb, ka, kb] == (pow(generator, ka << (ja * window3a), period) << (jb * window3b)) * kb % period
        table3b[i, ja, jb, ka, kb] == (-pow(generator, -ka << (ja * window3a), period) << (jb * window3b)) * kb % period
        table3c[i, k] == pow(generator, k, period)

    The format of the files is straight binary data: a series of 32 or 64 bit unsigned integers in
    the native byte endian-ness of the system that generated it. Here is the format of the file,
    written as a C struct:

        #if rns_primes_bit_length <= 32
            #define period_t uint32_t
        #else
            #define period_t uint64_t
        #endif
        struct Table3aDataFormat {
            period_t offsets[num_periods][num_windows2a][num_windows2b][1 << window3a][1 << window3b];
        }
        struct Table3bDataFormat {
            period_t offsets[num_periods][num_windows2a][num_windows2b][1 << window3a][1 << window3b];
        }
        struct Table3cDataFormat {
            period_t initial_values[num_periods][1 << (2 * window3a)];
        }

    Here is how to read the table3a and table3b files using numpy:

        >>> period_t = np.uint32 if rns_primes_bit_length <= 32 else np.uint64
        >>> shape3ab = (len(periods), num_windows2a, num_windows2b, 1 << (window3a + window3b))
        >>> shape3c = (len(periods), 1 << (window3a * 2))
        >>> table3a = np.fromfile('table3a.dat', count=math.prod(shape3ab), dtype=period_t)
        >>> table3b = np.fromfile('table3b.dat', count=math.prod(shape3ab), dtype=period_t)
        >>> table3c = np.fromfile('table3c.dat', count=math.prod(shape3c), dtype=period_t)
        >>> table3a = table3a.reshape(shape3ab)
        >>> table3b = table3b.reshape(shape3ab)
        >>> table3c = table3c.reshape(shape3c)

    Note that the size variables are listed in `metadata.ini`.


=== table4.dat ===
    This file lists values to add into the approximate accumulator to account for a residue's contributions.

    The table values satisfy:

        table4[i, j, k] = ((U[i] << (j * window4)) * k % T % N) >> dropped_bits

        where N is the number being factored
        where T is the product of all periods in the residue number system
        where U[i] = (T // periods[i]) * pow(T // periods[i], -1, periods[i]) are the one-hot residue values

    The format of the file is straight binary data: a series of 64 bit unsigned integers in
    the native byte endian-ness of the system that generated it. Here is the format of the file,
    written as a C struct:

        struct Table4DataFormat {
            uint64_t approx_offsets[num_periods][num_windows3_approx_offsets][1 << window4];
        }

    And here is how to read the file using numpy:

        >>> shape4 = (len(periods), conf.num_windows3_approx_offsets, 1 << conf.window4)
        >>> table4 = np.fromfile('table4.dat', count=math.prod(shape4), dtype=np.uint64)
        >>> table4.reshape(shape4)

    Note that the size variables are listed in `metadata.ini`.
"""


def black_box_costs(
    costs: Mapping[CostKey, int], reaction_us: int, clifford_us: int
) -> collections.Counter[str]:
    result = collections.Counter()
    for k, v in costs.items():
        if k.name == "__iadd__":
            tof = k["n"] - 2
            us = 2 * (k["n"] - 1) * reaction_us + clifford_us
        elif k.name == "phase_flip_if_cmp":
            tof = k["n"]
            us = reaction_us * 2 * k["n"] + clifford_us
        elif k.name == "init_lookup":
            N = k["N"]
            tof = N - N.bit_length() - 1
            us = N * clifford_us
        elif k.name == "phaseflip_by_lookup":
            N = k["N"]
            tof = math.isqrt(N)
            us = 2 * math.isqrt(N) * clifford_us
        elif k.name == "CX":
            continue
        elif k.name == "CZ":
            continue
        else:
            raise NotImplementedError(f"{k=}")
        result["toffolis"] += tof * v
        result["microseconds"] += us * v
    result["toffolis"] = round(result["toffolis"])
    result["microseconds"] = round(result["microseconds"])
    return result


def simplified_costs(costs: Mapping[CostKey, int]) -> collections.Counter[CostKey]:
    seen_adder_sizes = set()
    seen_qrom_sizes = set()
    seen_phase_lookup_sizes = set()
    adders = 0
    qroms = 0
    phase_lookups = 0
    for k, v in costs.items():
        if k.name == "__iadd__" or k.name == "phase_flip_if_cmp":
            seen_adder_sizes.add(k["n"])
            adders += v
        elif k.name == "init_lookup":
            if k["N"] <= 2:
                continue  # Too cheap to meter.
            seen_qrom_sizes.add(k["N"])
            qroms += v
        elif k.name == "phaseflip_by_lookup":
            if k["N"] <= 4:
                continue  # Too cheap to meter.
            seen_phase_lookup_sizes.add(k["N"])
            phase_lookups += v
        elif k.name == "CX":
            pass
        elif k.name == "CZ":
            pass
        else:
            raise NotImplementedError(f"{k=}")
    result: collections.Counter[CostKey] = collections.Counter()
    if adders:
        result[CostKey("add", {"n": max(seen_adder_sizes)})] = adders
    if qroms:
        result[CostKey("qrom", {"N": max(seen_qrom_sizes)})] = qroms
    if phase_lookups:
        result[CostKey("phase_lookup", {"N": max(seen_phase_lookup_sizes)})] = phase_lookups
    return result
