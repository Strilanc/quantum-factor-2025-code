from __future__ import annotations

import dataclasses
import functools
import math
import pathlib
from typing import Iterable, Iterator, Literal, cast

from facto.algorithm.prep._residue_util import prime_count_and_capacity_at_bit_length


@dataclasses.dataclass(frozen=True, unsafe_hash=True)
class ProblemConfig:
    """A description of a factoring problem, alongside algorithm configuration parameters.

    Args:
        modulus: The number to factor (N).
        num_input_qubits: The number of qubits in the exponent register (e). Shor's algorithm
            traditionally uses 2*len(modulus) qubits, but there are variations on this.
        generator: The classical random value G in the function f(e) = pow(G, e, N) that Shor's
            algorithm performs period finding against.
        len_accumulator: Length of total accumulator register. The one that will contain the
            approximate modular exponentation.
        mask_bits: Number of qubits used for superposition masking.

        window1: Window size over the exponent register, used when accumulating discrete logs
            of a residue's masked product.
        window3a: Window size over the exponent register, used when converting the discrete log into
            normal value.
        window3b: Window size over the residue discrete log register, used when converting the
            discrete log of the residue into a normal residue.
        window4: Window size over the residue register, used when adding its
            contributions into the total accumulator register.
        min_wraparound_gap: A constraint on the residue number system's modulus (L) versus the
            number to factor (N). It must be the case that L < (N >> min_wraparound_gap). Bounds
            the modular deviation that can result from not counting wraparounds mod L.

        rns_primes_bit_length: Optional (defaults to autosolve). Specifies the desired bit length of
            primes in the residue number system.
        rns_primes_range_start: Optional (defaults to autosolve). Specifies the (inclusive) start of
            a range of primes to include in the residue number system.
        rns_primes_range_stop: Optional (defaults to autosolve). Specifies the (exclusive) end of a
            range of primes to include in the residue number system.
        rns_primes_skipped: Optional (defaults to autosolve). A list of primes, from the start:stop
            range, to omit even though they were in the range. Primes can be omitted for various
            reasons, such as avoiding annoying corner cases like multiplying by 0.
        rns_primes_extra: Optional (defaults to autosolve). A list of primes, outside the start:stop
            range, to include even though they weren't in the range.

        parallelism: Optional (defaults to 1). Specifies how many residues the algorithm should be
            computing in parallel, paying space to save time.
        num_shots: Optional (defaults to 1). Specifies, for making estimates, the expected number of
            times the algorithm will run.

    """

    modulus: int
    num_input_qubits: int
    generator: int
    mask_bits: int

    window1: int
    window3a: int
    window3b: int
    window4: int
    min_wraparound_gap: int
    len_accumulator: int
    parallelism: int
    num_shots: int
    pp_success_probability: float
    rns_primes_bit_length: int | None
    rns_primes_range_start: int | None
    rns_primes_range_stop: int | None
    rns_primes_extra: tuple[int, ...]
    rns_primes_skipped: tuple[int, ...]

    def __post_init__(self):
        if not isinstance(self.modulus, int):
            raise TypeError(f"not isinstance({self.modulus=}, int)")
        if not isinstance(self.window1, int):
            raise TypeError(f"not isinstance({self.window1=}, int)")
        if not isinstance(self.window3a, int):
            raise TypeError(f"not isinstance({self.window3a=}, int)")
        if not isinstance(self.window3b, int):
            raise TypeError(f"not isinstance({self.window3b=}, int)")
        if not isinstance(self.window4, int):
            raise TypeError(f"not isinstance({self.window4=}, int)")
        if not isinstance(self.min_wraparound_gap, int):
            raise TypeError(f"not isinstance({self.min_wraparound_gap=}, int)")
        if not isinstance(self.len_accumulator, int):
            raise TypeError(f"not isinstance({self.len_accumulator=}, int)")
        if not isinstance(self.num_input_qubits, int):
            raise TypeError(f"not isinstance({self.num_input_qubits=}, int)")
        if not isinstance(self.generator, int):
            raise TypeError(f"not isinstance({self.generator=}, int)")
        if not isinstance(self.parallelism, int):
            raise TypeError(f"not isinstance({self.parallelism=}, int)")
        if not isinstance(self.num_shots, int):
            raise TypeError(f"not isinstance({self.num_shots=}, int)")
        if not isinstance(self.pp_success_probability, float):
            raise TypeError(f"not isinstance({self.pp_success_probability=}, float)")
        if not isinstance(self.rns_primes_bit_length, int | None):
            raise TypeError(f"not isinstance({self.rns_primes_bit_length=}, int)")
        if not isinstance(self.rns_primes_range_start, int | None):
            raise TypeError(f"not isinstance({self.rns_primes_range_start=}, int)")
        if not isinstance(self.rns_primes_range_stop, int | None):
            raise TypeError(f"not isinstance({self.rns_primes_range_stop=}, int)")
        if not isinstance(self.rns_primes_extra, tuple | None):
            raise TypeError(f"not isinstance({self.rns_primes_extra=}, tuple)")
        if not isinstance(self.rns_primes_skipped, tuple | None):
            raise TypeError(f"not isinstance({self.rns_primes_skipped=}, tuple)")

    def with_edits(
        self,
        *,
        rns_primes_bit_length: int | None | Literal["keep"] = "keep",
        parallelism: int | Literal["keep"] = "keep",
        len_accumulator: int | Literal["keep"] = "keep",
        window1: int | Literal["keep"] = "keep",
        window3a: int | Literal["keep"] = "keep",
        window3b: int | Literal["keep"] = "keep",
        window4: int | Literal["keep"] = "keep",
        rns_primes_range_start: int | None | Literal["keep"] = "keep",
        rns_primes_range_stop: int | None | Literal["keep"] = "keep",
        rns_primes_extra: tuple[int, ...] | Literal["keep"] = "keep",
        rns_primes_skipped: tuple[int, ...] | Literal["keep"] = "keep",
        mask_bits: int | Literal["keep"] = "keep",
    ) -> ProblemConfig:
        return ProblemConfig(
            modulus=self.modulus,
            mask_bits=cast(int, mask_bits) if mask_bits != "keep" else self.mask_bits,
            window1=cast(int, window1 if window1 != "keep" else self.window1),
            window3a=cast(int, window3a if window3a != "keep" else self.window3a),
            window3b=cast(int, window3b if window3b != "keep" else self.window3b),
            window4=cast(int, window4 if window4 != "keep" else self.window4),
            min_wraparound_gap=self.min_wraparound_gap,
            len_accumulator=cast(
                int, len_accumulator if len_accumulator != "keep" else self.len_accumulator
            ),
            rns_primes_bit_length=cast(
                int | None,
                (
                    rns_primes_bit_length
                    if rns_primes_bit_length != "keep"
                    else self.rns_primes_bit_length
                ),
            ),
            rns_primes_range_start=cast(
                int | None,
                (
                    rns_primes_range_start
                    if rns_primes_range_start != "keep"
                    else self.rns_primes_range_start
                ),
            ),
            rns_primes_range_stop=cast(
                int | None,
                (
                    rns_primes_range_stop
                    if rns_primes_range_stop != "keep"
                    else self.rns_primes_range_stop
                ),
            ),
            rns_primes_extra=cast(
                tuple[int, ...] | None,
                rns_primes_extra if rns_primes_extra != "keep" else self.rns_primes_extra,
            ),
            rns_primes_skipped=cast(
                tuple[int, ...] | None,
                rns_primes_skipped if rns_primes_skipped != "keep" else self.rns_primes_skipped,
            ),
            parallelism=cast(int, parallelism if parallelism != "keep" else self.parallelism),
            num_input_qubits=self.num_input_qubits,
            generator=self.generator,
            num_shots=self.num_shots,
            pp_success_probability=self.pp_success_probability
        )

    def estimate_minimum_rns_period_bit_length(self) -> int:
        """Uses the prime number theorem to estimate the needed size of prime.

        There need to be enough primes, of a large enough size, to represent N**n1
        where N is the number to factor and n1 is the number of values that might be
        multiplied together (the size of the mask/exponent divided by the window size).
        """

        # These are to ensure a large search space of prime subsets.
        safety_size = 8
        safety_factor = 1.1

        max_product_bits = self.modulus.bit_length() * self.num_windows1
        estimated_bit_length = math.ceil(safety_factor * math.log2(max_product_bits * math.log(4)))
        while prime_count_and_capacity_at_bit_length(estimated_bit_length)[1] >= max_product_bits:
            estimated_bit_length -= 1
        while (
            prime_count_and_capacity_at_bit_length(estimated_bit_length)[1]
            < (max_product_bits + estimated_bit_length * 100) * safety_factor
        ):
            estimated_bit_length += 1

        return max(safety_size, estimated_bit_length)

    @functools.cached_property
    def len_dlog_accumulator(self) -> int:
        """Size of the accumulator for adding up discrete logs of the residue."""
        assert self.rns_primes_bit_length is not None
        return self.rns_primes_bit_length + self.num_input_qubits.bit_length()

    @functools.cached_property
    def truncated_modulus(self) -> int:
        """Modulus used for the truncated output register."""
        return self.modulus >> self.dropped_bits

    @functools.cached_property
    def num_windows1(self) -> int:
        return -(-self.num_input_qubits // self.window1)

    @functools.cached_property
    def num_windows3a(self) -> int:
        return -(-self.rns_primes_bit_length // self.window3a)

    @functools.cached_property
    def num_windows3b(self) -> int:
        return -(-self.rns_primes_bit_length // self.window3b)

    @functools.cached_property
    def num_windows4(self) -> int:
        assert self.rns_primes_bit_length is not None
        return -(-self.rns_primes_bit_length // self.window4)

    @functools.cached_property
    def dropped_bits(self) -> int:
        return max(0, self.modulus.bit_length() - self.len_accumulator)

    @staticmethod
    def from_ini_content(lines: str | Iterator[str] | Iterable[str]) -> ProblemConfig:
        if isinstance(lines, str):
            lines = lines.splitlines()
        kv: dict[str, int | tuple[int, ...]] = {}
        for line in lines:
            if ";" in line:
                line = line[: line.index(";")]
            line = line.strip()
            if not line:
                continue
            terms = line.split("=")
            if len(terms) != 2:
                raise ValueError(f"Don't know how to parse {line=}")
            key, val = terms
            key = key.strip()
            val = val.strip()
            if val.startswith("[") and val.endswith("]"):
                if val == "[]":
                    val = ()
                else:
                    val = tuple(int(e) for e in val[1:-1].split(","))
            else:
                val = int(val)
            if key in kv:
                raise ValueError(f"Duplicate {key=}")
            kv[key] = val

        result = ProblemConfig(
            modulus=kv.pop("modulus"),
            window1=kv.pop("window1"),
            window3a=kv.pop("window3a"),
            window3b=kv.pop("window3b"),
            window4=kv.pop("window4"),
            mask_bits=kv.pop("mask_bits"),
            min_wraparound_gap=kv.pop("min_wraparound_gap"),
            len_accumulator=kv.pop("len_accumulator"),
            rns_primes_bit_length=kv.pop("rns_primes_bit_length", None),
            rns_primes_range_start=kv.pop("rns_primes_range_start", None),
            rns_primes_range_stop=kv.pop("rns_primes_range_stop", None),
            rns_primes_extra=kv.pop("rns_primes_extra", None),
            rns_primes_skipped=kv.pop("rns_primes_skipped", None),
            num_input_qubits=kv.pop("num_input_qubits"),
            generator=kv.pop("generator"),
            parallelism=kv.pop("parallelism", 1),
            num_shots=kv.pop("num_shots"),
            pp_success_probability=kv.pop("pp_success_probability")
        )
        if kv:
            raise ValueError(f"Unrecognized keys: {sorted(kv.keys())}")
        return result

    @staticmethod
    def from_ini_path(path: str | pathlib.Path) -> ProblemConfig:
        with open(path, encoding="utf8") as f:
            return ProblemConfig.from_ini_content(f)

    def __str__(self) -> str:
        key_vals = [
            ("modulus", self.modulus),
            ("num_input_qubits", self.num_input_qubits),
            ("generator", self.generator),
            ("len_accumulator", self.len_accumulator),
            ("min_wraparound_gap", self.min_wraparound_gap),
            ("mask_bits", self.mask_bits),
            "",
            ("window1", self.window1),
            ("window3a", self.window3a),
            ("window3b", self.window3b),
            ("window4", self.window4),
            ("parallelism", self.parallelism),
            ("num_shots", self.num_shots),
            ("pp_success_probability", self.pp_success_probability),
            "",
            ("rns_primes_bit_length", self.rns_primes_bit_length),
            ("rns_primes_range_start", self.rns_primes_range_start),
            ("rns_primes_range_stop", self.rns_primes_range_stop),
            ("rns_primes_extra", self.rns_primes_extra),
            ("rns_primes_skipped", self.rns_primes_skipped),
        ]
        lines = []
        for key_val in key_vals:
            if key_val == "":
                lines.append("")
            else:
                key, val = key_val
                if val is None:
                    pass
                elif isinstance(val, int):
                    lines.append(f"{key} = {val}")
                elif isinstance(val, tuple):
                    lines.append(f"{key} = {list(val)}")
                else:
                    raise NotImplementedError(f"{key_val=}")
        return "\n".join(lines)
