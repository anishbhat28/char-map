from collections import defaultdict, deque
import numpy as np

from .types import Prediction


class Policy:
    name = "base"
    def reset(self): pass
    def observe_requests(self, requests): pass
    def predict(self, **kwargs): return []


class ReactivePolicy(Policy):
    name = "reactive"


def _window_predictions(center, pe, N, window_k, needed_at):
    radius = window_k // 2
    out = []
    for off in range(-radius, radius + 1):
        out.append(
            Prediction(
                (center + off) % N,
                pe,
                needed_at,
                1.0 / (1 + abs(off)),
            )
        )
    return out


class VelocityHistoryPolicy(Policy):
    def __init__(self, window_k=1):
        self.window_k = window_k
        self.name = f"history_velocity_k{window_k}"
        self.history = defaultdict(lambda: deque(maxlen=2))

    def reset(self):
        self.history = defaultdict(lambda: deque(maxlen=2))

    def observe_requests(self, requests):
        for r in requests:
            self.history[r.consumer_pe].append(r.block_id)

    @staticmethod
    def cyclic_delta(a, b, N):
        return min([b-a, b-a+N, b-a-N], key=abs)

    def predict(
        self,
        *,
        physics,
        timestep,
        cycle,
        horizon,
        compute_latency,
    ):
        out = []
        for pe in range(physics.num_pes):
            h = self.history[pe]
            if not h:
                continue

            if len(h) == 1:
                center = h[-1]
            else:
                delta = self.cyclic_delta(
                    h[-2],
                    h[-1],
                    physics.N,
                )
                center = (
                    h[-1]
                    + (horizon + 1) * delta
                ) % physics.N

            out += _window_predictions(
                int(center),
                pe,
                physics.N,
                self.window_k,
                cycle + horizon * compute_latency,
            )
        return out


class PolynomialHistoryPolicy(Policy):
    def __init__(
        self,
        history_len=6,
        degree=2,
        window_k=1,
    ):
        self.history_len = history_len
        self.degree = degree
        self.window_k = window_k
        self.name = f"history_poly_k{window_k}"
        self.history = defaultdict(
            lambda: deque(maxlen=history_len)
        )

    def reset(self):
        self.history = defaultdict(
            lambda: deque(maxlen=self.history_len)
        )

    def observe_requests(self, requests):
        for r in requests:
            self.history[r.consumer_pe].append(
                r.block_id
            )

    @staticmethod
    def unwrap(vals, N):
        if not vals:
            return []
        out = [float(vals[0])]
        for v in vals[1:]:
            candidates = [
                float(v) + k*N
                for k in (-2,-1,0,1,2)
            ]
            out.append(
                min(
                    candidates,
                    key=lambda x: abs(x-out[-1]),
                )
            )
        return out

    def predict(
        self,
        *,
        physics,
        timestep,
        cycle,
        horizon,
        compute_latency,
    ):
        out = []
        for pe in range(physics.num_pes):
            h = list(self.history[pe])
            if not h:
                continue

            if len(h) < 3:
                center = h[-1]
            else:
                y = np.array(
                    self.unwrap(h, physics.N),
                    dtype=float,
                )
                x = np.arange(
                    len(y),
                    dtype=float,
                )
                deg = min(
                    self.degree,
                    len(y)-1,
                )
                coeff = np.polyfit(
                    x,
                    y,
                    deg,
                )
                xq = (
                    len(y)-1
                    + horizon + 1
                )
                center = int(
                    round(
                        np.polyval(
                            coeff,
                            xq,
                        )
                    )
                ) % physics.N

            out += _window_predictions(
                center,
                pe,
                physics.N,
                self.window_k,
                cycle + horizon * compute_latency,
            )
        return out


class CharacteristicWindowPolicy(Policy):
    def __init__(
        self,
        window_k=1,
        amplitude_error=0.0,
        velocity_error=0.0,
        parameter_delay_steps=0,
    ):
        self.window_k = window_k
        self.amplitude_error = amplitude_error
        self.velocity_error = velocity_error
        self.parameter_delay_steps = parameter_delay_steps
        self.name = (
            f"physics_k{window_k}"
            f"_err{amplitude_error:g}"
            f"_delay{parameter_delay_steps}"
        )

    def predict(
        self,
        *,
        physics,
        timestep,
        cycle,
        horizon,
        compute_latency,
    ):
        out = []
        qstep = timestep + horizon + 1
        c_est = physics.c * (
            1.0 + self.velocity_error
        )

        for pe in range(physics.num_pes):
            center = physics.source_index(
                pe % physics.N,
                horizon=qstep,
                velocity_override=c_est,
                amplitude_scale=(
                    1.0 + self.amplitude_error
                ),
                parameter_delay_steps=(
                    self.parameter_delay_steps
                ),
            )

            out += _window_predictions(
                center,
                pe,
                physics.N,
                self.window_k,
                cycle + horizon * compute_latency,
            )

        return out


class OracleWindowPolicy(Policy):
    def __init__(self, window_k=1):
        self.window_k = window_k
        self.name = f"oracle_k{window_k}"

    def predict(
        self,
        *,
        physics,
        timestep,
        cycle,
        horizon,
        compute_latency,
    ):
        out = []
        qstep = timestep + horizon + 1

        for pe in range(physics.num_pes):
            center = physics.source_index(
                pe % physics.N,
                horizon=qstep,
                amplitude_scale=1.0,
                parameter_delay_steps=0,
            )

            out += _window_predictions(
                center,
                pe,
                physics.N,
                self.window_k,
                cycle + horizon * compute_latency,
            )

        return out
