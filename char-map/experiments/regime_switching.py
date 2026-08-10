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
PHYSICS_ERROR = 0.02

def main():
    cfg = sim_config_from_dict(
        BASE,
        overrides={
            "physics_model": "regime",
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
        UncertaintyCharacteristicPolicy(
            window_k=3,
            amplitude_error=PHYSICS_ERROR,
        ),
        OraclePolicy(),
    ]

    summaries = []
    per_timestep = []

    for p in policies:
        df, s = Simulator(cfg, p).run()

        summaries.append(s)

        d = (
            df.groupby("timestep", as_index=False)
              .agg(
                  max_stall=("stall_cycles", "max"),
                  mean_stall=("stall_cycles", "mean"),
                  hit_rate=("hit", "mean"),
              )
        )

        d["policy"] = p.name
        per_timestep.append(d)

    summary = pd.DataFrame(summaries)

    reactive_cycles = float(
        summary.loc[
            summary["policy"] == "reactive",
            "total_cycles",
        ].iloc[0]
    )

    summary["speedup_vs_reactive"] = (
        reactive_cycles / summary["total_cycles"]
    )

    per_t = pd.concat(
        per_timestep,
        ignore_index=True,
    )

    T = cfg.timesteps
    s1 = T // 3
    s2 = 2 * T // 3

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "plots").mkdir(exist_ok=True)

    summary.to_csv(
        ROOT / "results/regime_switch_summary.csv",
        index=False,
    )

    per_t.to_csv(
        ROOT / "results/regime_switch_timestep.csv",
        index=False,
    )

    for p in per_t["policy"].unique():
        d = per_t[
            per_t["policy"] == p
        ]

        plt.plot(
            d["timestep"],
            d["max_stall"],
            label=p,
        )

    plt.axvline(
        s1,
        linestyle="--",
    )

    plt.axvline(
        s2,
        linestyle="--",
    )

    plt.xlabel("Timestep")
    plt.ylabel("Critical-path stall cycles")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        ROOT / "plots/regime_switch_stalls.png",
        dpi=180,
    )

    plt.close()

    recovery_rows = []

    for switch_name, switch_t in [
        ("switch_1", s1),
        ("switch_2", s2),
    ]:
        for p in per_t["policy"].unique():

            d = per_t[
                (per_t["policy"] == p)
                & (per_t["timestep"] >= switch_t)
                & (per_t["timestep"] < switch_t + 10)
            ]

            recovery_rows.append(
                {
                    "switch": switch_name,
                    "policy": p,
                    "avg_max_stall_first_10":
                        d["max_stall"].mean(),
                    "max_stall_first_10":
                        d["max_stall"].max(),
                    "avg_hit_rate_first_10":
                        d["hit_rate"].mean(),
                }
            )

    recovery = pd.DataFrame(
        recovery_rows
    )

    recovery.to_csv(
        ROOT / "results/regime_switch_recovery.csv",
        index=False,
    )

    print("\n=== Overall ===")

    print(
        summary[
            [
                "policy",
                "total_cycles",
                "hit_rate",
                "bytes_transferred",
                "speedup_vs_reactive",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\n=== First 10 timesteps after each switch ==="
    )

    print(
        recovery.to_string(
            index=False
        )
    )

if __name__ == "__main__":
    main()
