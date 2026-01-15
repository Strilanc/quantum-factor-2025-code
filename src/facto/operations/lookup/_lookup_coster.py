from __future__ import annotations

import dataclasses
import functools


@dataclasses.dataclass(frozen=True)
class LookupCoster:
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
    def and_gates_for_power_product(self) -> int:
        m1, m2 = self.power_product_split
        p1 = (1 << m1) - 1
        p2 = (1 << m2) - 1
        return (p1 - m1) + (p2 - m2)

    @functools.cached_property
    def multi_target_ccx_gates(self) -> int:
        m1, m2 = self.power_product_split
        p1 = (1 << m1) - 1
        p2 = (1 << m2) - 1
        return p1 * p2

    @functools.cached_property
    def multi_target_cx_gates(self) -> int:
        result = (1 << self.addr_len) - 1 - self.multi_target_ccx_gates
        m1, m2 = self.power_product_split
        p1 = (1 << m1) - 1
        p2 = (1 << m2) - 1
        assert result == p1 + p2
        return result

    @functools.cached_property
    def total_ccz_states(self) -> int:
        result = self.and_gates_for_power_product + self.multi_target_ccx_gates
        assert result == (1 << self.addr_len) - self.addr_len - 1
        return result

    @functools.cached_property
    def reaction_depth(self) -> int:
        return self.multi_target_ccx_gates + (self.addr_len > 2)

    @functools.cached_property
    def workspace(self) -> int:
        m1, m2 = self.power_product_split
        return ((1 << m1) - m1 - 1) + ((1 << m2) - m2 - 1)

    @functools.cached_property
    def lookup_duration_micros(self) -> float:
        power_product_and_gates = self.and_gates_for_power_product
        multi_target_ccx_gates = self.multi_target_ccx_gates
        multi_target_cx_gates = self.multi_target_cx_gates
        R = self.reaction_micros
        S = self.surgery_micros
        M = self.factory_micros / self.factory_count
        return (
            1 * R
            + power_product_and_gates * M
            + multi_target_ccx_gates * max(M, S, R)
            + multi_target_cx_gates * S
        )
