from __future__ import annotations

import dataclasses
import functools
import math


@dataclasses.dataclass(frozen=True)
class AddCoster:
    n: int
    surgery_micros: float
    reaction_micros: float
    factory_micros: float
    factory_count: int

    @functools.cached_property
    def adder_compute_layers(self) -> float:
        return math.ceil(self.n / self.factory_count) * 6

    @functools.cached_property
    def adder_uncompute_layers(self) -> float:
        return math.ceil(self.n / self.factory_count) * 3

    @functools.cached_property
    def adder_ccz_states(self) -> float:
        return self.n - 1

    @functools.cached_property
    def adder_duration_micros(self) -> float:
        compute_layers = self.adder_compute_layers
        uncompute_layers = self.adder_uncompute_layers
        ccz_states = self.adder_ccz_states
        compute_time = max(
            # Surgery limited.
            compute_layers * self.surgery_micros,
            # Magic limited.
            ccz_states *  self.factory_micros / self.factory_count,
            # Reaction limited.
            ccz_states * self.reaction_micros,
        )
        uncompute_time = max(
            # Surgery limited.
            uncompute_layers * self.surgery_micros,
            # Reaction limited.
            ccz_states * self.reaction_micros,
        )
        return compute_time + uncompute_time
