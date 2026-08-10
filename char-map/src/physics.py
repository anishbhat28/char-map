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
    def dx(self):
        return self.L / self.N

    def characteristic(self, x, horizon=1, velocity_override=None):
        c = self.c if velocity_override is None else velocity_override
        return (x - c * horizon * self.dt) % self.L

    def source_index(
        self,
        cell_i,
        horizon=1,
        velocity_override=None,
        amplitude_scale=1.0,
        parameter_delay_steps=0,
    ):
        z = self.characteristic(
            cell_i * self.dx,
            horizon,
            velocity_override,
        )
        return int(round(z / self.dx)) % self.N

    def owner_pe(self, block_id):
        return block_id % self.num_pes

    def requests_at(self, timestep, current_cycle):
        out = []
        for i in range(self.N):
            b = self.source_index(i, horizon=timestep + 1)
            out.append(
                Request(
                    timestep,
                    i % self.num_pes,
                    b,
                    self.owner_pe(b),
                    current_cycle,
                )
            )
        return out


@dataclass
class VariableAdvection:
    c0: float
    amplitude: float
    omega: float
    L: float
    N: int
    dt: float
    num_pes: int
    rk4_substeps_per_dt: int = 8

    @property
    def dx(self):
        return self.L / self.N

    @property
    def c(self):
        return self.c0

    def amplitude_at(self, t):
        return self.amplitude

    def _effective_amplitude(self, t, parameter_delay_steps=0):
        stale_t = max(0.0, t - parameter_delay_steps * self.dt)
        return self.amplitude_at(stale_t)

    def velocity(
        self,
        x,
        t,
        scale=1.0,
        amplitude_scale=1.0,
        parameter_delay_steps=0,
    ):
        amp = self._effective_amplitude(
            t,
            parameter_delay_steps=parameter_delay_steps,
        )
        return scale * (
            self.c0
            + amplitude_scale
            * amp
            * math.sin(2.0 * math.pi * (x % self.L) / self.L)
            * math.cos(self.omega * t)
        )

    def _rk4_step(
        self,
        x,
        t,
        h,
        velocity_scale=1.0,
        amplitude_scale=1.0,
        parameter_delay_steps=0,
    ):
        def f(xx, tt):
            return self.velocity(
                xx,
                tt,
                scale=velocity_scale,
                amplitude_scale=amplitude_scale,
                parameter_delay_steps=parameter_delay_steps,
            )

        k1 = f(x, t)
        k2 = f(x + 0.5 * h * k1, t + 0.5 * h)
        k3 = f(x + 0.5 * h * k2, t + 0.5 * h)
        k4 = f(x + h * k3, t + h)

        return (
            x + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        ) % self.L

    def characteristic(
        self,
        x_query,
        query_step,
        velocity_scale=1.0,
        amplitude_scale=1.0,
        parameter_delay_steps=0,
    ):
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
                parameter_delay_steps=parameter_delay_steps,
            )
            t += h

        return x % self.L

    def source_index(
        self,
        cell_i,
        horizon=1,
        velocity_override=None,
        amplitude_scale=1.0,
        parameter_delay_steps=0,
    ):
        velocity_scale = (
            1.0
            if velocity_override is None
            else velocity_override / self.c0
        )

        z = self.characteristic(
            x_query=cell_i * self.dx,
            query_step=horizon,
            velocity_scale=velocity_scale,
            amplitude_scale=amplitude_scale,
            parameter_delay_steps=parameter_delay_steps,
        )

        return int(round(z / self.dx)) % self.N

    def owner_pe(self, block_id):
        return block_id % self.num_pes

    def requests_at(self, timestep, current_cycle):
        # Ground truth is always exact/current physics.
        out = []
        qstep = timestep + 1
        for i in range(self.N):
            b = self.source_index(
                i,
                horizon=qstep,
                amplitude_scale=1.0,
                parameter_delay_steps=0,
            )
            out.append(
                Request(
                    timestep,
                    i % self.num_pes,
                    b,
                    self.owner_pe(b),
                    current_cycle,
                )
            )
        return out


@dataclass
class RegimeSwitchingAdvection(VariableAdvection):
    amplitude_1: float = 0.25
    amplitude_2: float = 0.75
    amplitude_3: float = 0.40
    switch_1: float = 1.0 / 3.0
    switch_2: float = 2.0 / 3.0
    total_steps: int = 100

    def amplitude_at(self, t):
        total_time = max(self.dt, self.total_steps * self.dt)
        frac = t / total_time
        if frac < self.switch_1:
            return self.amplitude_1
        if frac < self.switch_2:
            return self.amplitude_2
        return self.amplitude_3
