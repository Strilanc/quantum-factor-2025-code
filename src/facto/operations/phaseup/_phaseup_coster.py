from __future__ import annotations

import dataclasses
import functools

from facto.operations.phaseup._phaseup_imp import estimate_cost_of_sqrt_phaseup
from scatter_script import CostKey


@dataclasses.dataclass(frozen=True)
class PhaseupCoster:
    addr_len: int
    surgery_micros: float
    reaction_micros: float
    factory_micros: float
    factory_count: int

    @functools.cached_property
    def power_product_split(self) -> tuple[int, int]:
        m1 = self.addr_len // 2
        m2 = self.addr_len - m1
        return m1, m2

    @functools.cached_property
    def reaction_depth(self) -> int:
        if self.addr_len < 2:
            return 0
        return (self.addr_len + 3) // 2

    @functools.cached_property
    def workspace(self) -> int:
        m1, m2 = self.power_product_split
        return ((1 << m1) - m1 - 1) + ((1 << m2) - m2 - 1)

    @functools.cached_property
    def multi_czs(self) -> int:
        e = estimate_cost_of_sqrt_phaseup(self.addr_len)
        return e[CostKey("CZ_multi_target")]

    @functools.cached_property
    def ands(self) -> int:
        e = estimate_cost_of_sqrt_phaseup(self.addr_len)
        return e[CostKey("uncorrected_and")]

    @functools.cached_property
    def phaseup_duration_micros(self) -> float:
        R = self.reaction_micros
        S = self.surgery_micros
        M = self.factory_micros / self.factory_count
        return (
                0
                # Computing power product
                # - negligible S dependence due to parallelism
                # - negligible R dependence due to wandering ANDs
                + self.ands * M
                # Performing actual phase operations.
                + self.multi_czs * S
                # Uncomputing power product.
                + self.addr_len * S
                + self.reaction_depth * R
        )
