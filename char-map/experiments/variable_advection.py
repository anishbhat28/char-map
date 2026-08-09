from pathlib import Path
import sys

ROOT = Path(
    __file__
).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT),
)

import pandas as pd
import matplotlib.pyplot as plt

from src.config import (
    load_yaml,
    sim_config_from_dict,
)

from src.simulator import (
    Simulator,
)

from src.policies import (
    ReactivePolicy,
    LastValuePolicy,
    VelocityHistoryPolicy,
    CharacteristicPolicy,
    OraclePolicy,
)

from src.metrics import (
    add_speedups,
)


BASE = load_yaml(
    ROOT / "configs/default.yaml"
)


def run_case(
    amplitude,
    omega,
):

    cfg = sim_config_from_dict(
        BASE,
        overrides={
            "physics_model":
                "variable",

            "variable_amplitude":
                amplitude,

            "variable_omega":
                omega,

            "rk4_substeps_per_dt":
                8,
        },
    )

    policies = [
        ReactivePolicy(),

        LastValuePolicy(),

        VelocityHistoryPolicy(),

        CharacteristicPolicy(
            velocity_error=0.0
        ),

        OraclePolicy(),
    ]

    rows = []

    for policy in policies:

        _, summary = Simulator(
            cfg,
            policy,
        ).run()

        rows.append(
            summary
        )

    df = add_speedups(
        pd.DataFrame(rows)
    )

    df["amplitude"] = amplitude
    df["omega"] = omega

    return df


def main():

    results = []

    amplitudes = [
        0.0,
        0.25,
        0.50,
        0.75,
        0.90,
    ]

    omega = 4.0

    for amplitude in amplitudes:

        df = run_case(
            amplitude,
            omega,
        )

        results.append(
            df
        )

    out = pd.concat(
        results,
        ignore_index=True,
    )

    result_dir = (
        ROOT / "results"
    )

    plot_dir = (
        ROOT / "plots"
    )

    result_dir.mkdir(
        exist_ok=True
    )

    plot_dir.mkdir(
        exist_ok=True
    )

    out.to_csv(
        result_dir
        / "variable_advection.csv",

        index=False,
    )

    for policy in (
        out["policy"].unique()
    ):

        d = out[
            out["policy"]
            == policy
        ]

        plt.plot(
            d["amplitude"],
            d[
                "speedup_vs_reactive"
            ],
            marker="o",
            label=policy,
        )

    plt.xlabel(
        "Velocity-field amplitude"
    )

    plt.ylabel(
        "Speedup vs reactive"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        plot_dir
        / "variable_advection.png",

        dpi=180,
    )

    plt.close()

    print(
        out[
            [
                "amplitude",
                "policy",
                "total_cycles",
                "hit_rate",
                "speedup_vs_reactive",
            ]
        ].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
