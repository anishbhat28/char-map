from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List
from .types import Request

@dataclass
class ConstantAdvection:
    c: float; L: float; N: int; dt: float; num_pes: int
    @property
    def dx(self): return self.L / self.N
    def characteristic(self, x, horizon=1, velocity_override=None):
        c = self.c if velocity_override is None else velocity_override
        return (x - c*horizon*self.dt) % self.L
    def source_index(self, cell_i, horizon=1, velocity_override=None, amplitude_scale=1.0):
        z = self.characteristic(cell_i*self.dx, horizon, velocity_override)
        return int(round(z/self.dx)) % self.N
    def owner_pe(self, block_id): return block_id % self.num_pes
    def requests_at(self, timestep, current_cycle):
        return [Request(timestep, i % self.num_pes, b, self.owner_pe(b), current_cycle)
                for i in range(self.N)
                for b in [self.source_index(i, horizon=timestep+1)]]

@dataclass
class VariableAdvection:
    c0: float; amplitude: float; omega: float; L: float; N: int; dt: float; num_pes: int
    rk4_substeps_per_dt: int = 8
    @property
    def dx(self): return self.L / self.N
    @property
    def c(self): return self.c0
    def velocity(self, x, t, scale=1.0, amplitude_scale=1.0):
        return scale*(self.c0 + amplitude_scale*self.amplitude*
                     math.sin(2*math.pi*(x % self.L)/self.L)*math.cos(self.omega*t))
    def _rk4_step(self, x, t, h, velocity_scale=1.0, amplitude_scale=1.0):
        f=lambda xx,tt:self.velocity(xx,tt,velocity_scale,amplitude_scale)
        k1=f(x,t); k2=f(x+.5*h*k1,t+.5*h); k3=f(x+.5*h*k2,t+.5*h); k4=f(x+h*k3,t+h)
        return (x+h*(k1+2*k2+2*k3+k4)/6) % self.L
    def characteristic(self, x_query, query_step, velocity_scale=1.0, amplitude_scale=1.0):
        if query_step <= 0: return x_query % self.L
        t=query_step*self.dt; x=x_query % self.L
        n=max(1,query_step*self.rk4_substeps_per_dt); h=-t/n
        for _ in range(n):
            x=self._rk4_step(x,t,h,velocity_scale,amplitude_scale); t+=h
        return x % self.L
    def source_index(self, cell_i, horizon=1, velocity_override=None, amplitude_scale=1.0):
        scale=1.0 if velocity_override is None else velocity_override/self.c0
        z=self.characteristic(cell_i*self.dx,horizon,scale,amplitude_scale)
        return int(round(z/self.dx)) % self.N
    def owner_pe(self, block_id): return block_id % self.num_pes
    def requests_at(self, timestep, current_cycle):
        return [Request(timestep, i % self.num_pes, b, self.owner_pe(b), current_cycle)
                for i in range(self.N)
                for b in [self.source_index(i,horizon=timestep+1,amplitude_scale=1.0)]]
