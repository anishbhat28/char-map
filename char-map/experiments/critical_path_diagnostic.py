from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import matplotlib.pyplot as plt

from src.config import load_yaml, sim_config_from_dict
from src.simulator import Simulator
from src.policies import (
    ReactivePolicy,
    VelocityHistoryPolicy,
    PolynomialHistoryPolicy,
    MarkovHistoryPolicy,
    UncertaintyCharacteristicPolicy,
    OraclePolicy,
)

BASE = load_yaml(ROOT / "configs/default.yaml")

def main():
    cfg = sim_config_from_dict(
        BASE,
        overrides={
            "physics_model": "variable",
            "variable_amplitude": 0.75,
            "variable_omega": 4.0,
            "rk4_substeps_per_dt": 8,
        },
    )

    policies = [
        ReactivePolicy(),
        VelocityHistoryPolicy(window_k=3),
        PolynomialHistoryPolicy(history_len=6, degree=2, window_k=3),
        MarkovHistoryPolicy(window_k=3),
        UncertaintyCharacteristicPolicy(window_k=3, amplitude_error=0.02),
        OraclePolicy(),
    ]

    frames = []

    for p in policies:
        df, _ = Simulator(cfg, p).run()

        by_t = (
            df.groupby("timestep", as_index=False)
              .agg(
                  max_stall=("stall_cycles", "max"),
                  mean_stall=("stall_cycles", "mean"),
                  hit_rate=("hit", "mean"),
                  prefetch_hit_rate=("prefetched_hit", "mean"),
              )
        )
        by_t["policy"] = p.name
        frames.append(by_t)

    out = pd.concat(frames, ignore_index=True)

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "plots").mkdir(exist_ok=True)

    out.to_csv(
        ROOT / "results/critical_path_diagnostic.csv",
        index=False,
    )

    for p in out["policy"].unique():
        d = out[out["policy"] == p]
        plt.plot(
            d["timestep"],
            d["max_stall"],
            label=p,
        )

    plt.xlabel("Timestep")
    plt.ylabel("Critical-path stall cycles")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        ROOT / "plots/critical_path_stalls.png",
        dpi=180,
    )
    plt.close()

    summary = (
        out.groupby("policy", as_index=False)
           .agg(
               avg_max_stall=("max_stall", "mean"),
               p95_max_stall=("max_stall", lambda x: x.quantile(0.95)),
               avg_mean_stall=("mean_stall", "mean"),
               avg_hit_rate=("hit_rate", "mean"),
           )
    )

    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()
