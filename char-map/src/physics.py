from __future__ import annotations

import math
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

    def characteristic(
        self,
        x: float,
        horizon: int = 1,
        velocity_override: float | None = None,
    ) -> float:
        c = self.c if velocity_override is None else velocity_override
        return (x - c * horizon * self.dt) % self.L

    def source_index(
        self,
        cell_i: int,
        horizon: int = 1,
        velocity_override: float | None = None,
        amplitude_scale: float = 1.0,
    ) -> int:
        x = cell_i * self.dx
        z = self.characteristic(
            x,
            horizon=horizon,
            velocity_override=velocity_override,
        )
        return int(round(z / self.dx)) % self.N

    def owner_pe(self, block_id: int) -> int:
        return block_id % self.num_pes

    def requests_at(self, timestep: int, current_cycle: int) -> List[Request]:
        reqs = []
        for cell_i in range(self.N):
            block = self.source_index(cell_i, horizon=timestep + 1)
            reqs.append(
                Request(
                    timestep,
                    cell_i % self.num_pes,
                    block,
                    self.owner_pe(block),
                    current_cycle,
                )
            )
        return reqs


@dataclass
class VariableAdvection:
    """
    u_t + c(x,t) u_x = 0
    c(x,t) = c0 + A sin(2*pi*x/L) cos(omega*t)
    """

    c0: float
    amplitude: float
    omega: float
    L: float
    N: int
    dt: float
    num_pes: int
    rk4_substeps_per_dt: int = 8

    @property
    def dx(self) -> float:
        return self.L / self.N

    @property
    def c(self) -> float:
        return self.c0

    def velocity(
        self,
        x: float,
        t: float,
        scale: float = 1.0,
        amplitude_scale: float = 1.0,
    ) -> float:
        x = x % self.L
        return scale * (
            self.c0
            + amplitude_scale
            * self.amplitude
            * math.sin(2.0 * math.pi * x / self.L)
            * math.cos(self.omega * t)
        )

    def _rk4_step(
        self,
        x: float,
        t: float,
        h: float,
        velocity_scale: float = 1.0,
        amplitude_scale: float = 1.0,
    ) -> float:
        def f(x_, t_):
            return self.velocity(
                x_,
                t_,
                scale=velocity_scale,
                amplitude_scale=amplitude_scale,
            )

        k1 = f(x, t)
        k2 = f(x + 0.5 * h * k1, t + 0.5 * h)
        k3 = f(x + 0.5 * h * k2, t + 0.5 * h)
        k4 = f(x + h * k3, t + h)

        return (
            x
            + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        ) % self.L

    def characteristic(
        self,
        x_query: float,
        query_step: int,
        velocity_scale: float = 1.0,
        amplitude_scale: float = 1.0,
    ) -> float:
        if query_step <= 0:
            return x_query % self.L

        t = query_step * self.dt
        x = x_query % self.L

        num_steps = max(
            1,
            query_step * self.rk4_substeps_per_dt,
        )

        h = -t / num_steps

        for _ in range(num_steps):
            x = self._rk4_step(
                x,
                t,
                h,
                velocity_scale=velocity_scale,
                amplitude_scale=amplitude_scale,
            )
            t += h

        return x % self.L

    def source_index(
        self,
        cell_i: int,
        horizon: int = 1,
        velocity_override: float | None = None,
        amplitude_scale: float = 1.0,
    ) -> int:
        x_query = cell_i * self.dx

        if velocity_override is None:
            velocity_scale = 1.0
        else:
            velocity_scale = velocity_override / self.c0

        z = self.characteristic(
            x_query=x_query,
            query_step=horizon,
            velocity_scale=velocity_scale,
            amplitude_scale=amplitude_scale,
        )

        return int(round(z / self.dx)) % self.N

    def owner_pe(self, block_id: int) -> int:
        return block_id % self.num_pes

    def requests_at(
        self,
        timestep: int,
        current_cycle: int,
    ) -> List[Request]:
        query_step = timestep + 1
        reqs = []

        for cell_i in range(self.N):
            block = self.source_index(
                cell_i,
                horizon=query_step,
                amplitude_scale=1.0,
            )

            reqs.append(
                Request(
                    timestep,
                    cell_i % self.num_pes,
                    block,
                    self.owner_pe(block),
                    current_cycle,
                )
            )

        return reqs


@dataclass
class RegimeSwitchingAdvection(VariableAdvection):
    """
    Same transport equation, but the velocity-field amplitude changes abruptly:
      first third: 0.25
      second third: 0.75
      final third: 0.40
    """

    amplitude_1: float = 0.25
    amplitude_2: float = 0.75
    amplitude_3: float = 0.40
    switch_1: float = 1.0 / 3.0
    switch_2: float = 2.0 / 3.0
    total_steps: int = 100

    def amplitude_at(self, t: float) -> float:
        total_time = max(self.dt, self.total_steps * self.dt)
        frac = t / total_time

        if frac < self.switch_1:
            return self.amplitude_1
        if frac < self.switch_2:
            return self.amplitude_2
        return self.amplitude_3

    def velocity(
        self,
        x: float,
        t: float,
        scale: float = 1.0,
        amplitude_scale: float = 1.0,
    ) -> float:
        x = x % self.L
        amp = self.amplitude_at(t)

        return scale * (
            self.c0
            + amplitude_scale
            * amp
            * math.sin(2.0 * math.pi * x / self.L)
            * math.cos(self.omega * t)
        )
