from __future__ import annotations
from dataclasses import dataclass
from typing import List
from .types import Request

@dataclass
class ConstantAdvection:
    c: float
    L: float
    N: int
    dt: float
    num_pes: int

    @property
    def dx(self) -> float:
        return self.L / self.N

    def characteristic(self, x: float, horizon: int = 1, velocity_override: float | None = None) -> float:
        c = self.c if velocity_override is None else velocity_override
        return (x - c * horizon * self.dt) % self.L

    def source_index(self, cell_i: int, horizon: int = 1, velocity_override: float | None = None) -> int:
        x = cell_i * self.dx
        z = self.characteristic(x, horizon=horizon, velocity_override=velocity_override)
        return int(round(z / self.dx)) % self.N

    def owner_pe(self, block_id: int) -> int:
        return block_id % self.num_pes

    def requests_at(self, timestep: int, current_cycle: int) -> List[Request]:
        reqs = []
        for cell_i in range(self.N):
            block = self.source_index(cell_i, horizon=timestep + 1)
            reqs.append(Request(timestep, cell_i % self.num_pes, block, self.owner_pe(block), current_cycle))
        return reqs
