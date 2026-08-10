# Add the following class to src/physics.py, below VariableAdvection.
# It reuses VariableAdvection's RK4 characteristic logic but switches amplitude
# across time. This file is a snippet, not a full replacement.

@dataclass
class RegimeSwitchingAdvection(VariableAdvection):
    switch_1: float = 0.33
    switch_2: float = 0.66
    amplitude_1: float = 0.25
    amplitude_2: float = 0.75
    amplitude_3: float = 0.40
    total_sim_time: float = 1.0

    def amplitude_at(self, t: float) -> float:
        frac = 0.0 if self.total_sim_time <= 0 else t / self.total_sim_time
        if frac < self.switch_1:
            return self.amplitude_1
        if frac < self.switch_2:
            return self.amplitude_2
        return self.amplitude_3

    def velocity(self, x, t, scale=1.0, amplitude_scale=1.0):
        amp = self.amplitude_at(t)
        return scale * (
            self.c0
            + amplitude_scale
            * amp
            * math.sin(2.0 * math.pi * (x % self.L) / self.L)
            * math.cos(self.omega * t)
        )
