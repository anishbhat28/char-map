from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import matplotlib.pyplot as plt

from src.config import (
    load_yaml,
    sim_config_from_dict,
)
from src.simulator import Simulator
from src.policies import (
    ReactivePolicy,
    VelocityHistoryPolicy,
    PolynomialHistoryPolicy,
    CharacteristicWindowPolicy,
    OracleWindowPolicy,
)

BASE = load_yaml(
    ROOT / "configs/default.yaml"
)

ERRORS = [
    0.00,
    0.01,
    0.02,
    0.05,
    0.10,
]

DELAYS = [
    0,
    1,
    2,
    4,
    8,
]

WINDOWS = [1, 3]


def make_cfg():
    return sim_config_from_dict(
        BASE,
        overrides={
            "physics_model": "regime",
            "variable_amplitude": 0.75,
            "variable_omega": 4.0,
            "rk4_substeps_per_dt": 8,
        },
    )


def run_one(policy):
    cfg = make_cfg()
    _, s = Simulator(
        cfg,
        policy,
    ).run()
    return s


def add_reference_metrics(df):
    reactive_cycles = float(
        df.loc[
            df["policy"] == "reactive",
            "total_cycles",
        ].iloc[0]
    )
    reactive_bytes = float(
        df.loc[
            df["policy"] == "reactive",
            "bytes_transferred",
        ].iloc[0]
    )

    df["speedup_vs_reactive"] = (
        reactive_cycles
        / df["total_cycles"]
    )

    df["traffic_overhead_vs_reactive"] = (
        df["bytes_transferred"]
        / reactive_bytes
    )

    return df


def equal_budget():
    rows = [
        run_one(
            ReactivePolicy()
        )
    ]

    for k in WINDOWS:
        rows += [
            run_one(
                VelocityHistoryPolicy(
                    window_k=k
                )
            ),
            run_one(
                PolynomialHistoryPolicy(
                    history_len=6,
                    degree=2,
                    window_k=k,
                )
            ),
            run_one(
                CharacteristicWindowPolicy(
                    window_k=k,
                )
            ),
            run_one(
                OracleWindowPolicy(
                    window_k=k,
                )
            ),
        ]

    out = add_reference_metrics(
        pd.DataFrame(rows)
    )
    out["experiment"] = (
        "equal_budget"
    )
    return out


def noisy_physics():
    rows = []

    for eps in ERRORS:
        # Each row bundle includes the same references.
        case = [
            run_one(ReactivePolicy()),
            run_one(
                VelocityHistoryPolicy(
                    window_k=3
                )
            ),
            run_one(
                PolynomialHistoryPolicy(
                    history_len=6,
                    degree=2,
                    window_k=3,
                )
            ),
            run_one(
                CharacteristicWindowPolicy(
                    window_k=3,
                    amplitude_error=eps,
                )
            ),
            run_one(
                OracleWindowPolicy(
                    window_k=3
                )
            ),
        ]

        df = add_reference_metrics(
            pd.DataFrame(case)
        )
        df["physics_error"] = eps
        rows.append(df)

    out = pd.concat(
        rows,
        ignore_index=True,
    )
    out["experiment"] = (
        "noisy_physics"
    )
    return out


def delayed_physics():
    rows = []

    for delay in DELAYS:
        case = [
            run_one(ReactivePolicy()),
            run_one(
                VelocityHistoryPolicy(
                    window_k=3
                )
            ),
            run_one(
                PolynomialHistoryPolicy(
                    history_len=6,
                    degree=2,
                    window_k=3,
                )
            ),
            run_one(
                CharacteristicWindowPolicy(
                    window_k=3,
                    parameter_delay_steps=delay,
                )
            ),
            run_one(
                OracleWindowPolicy(
                    window_k=3
                )
            ),
        ]

        df = add_reference_metrics(
            pd.DataFrame(case)
        )
        df["physics_delay"] = delay
        rows.append(df)

    out = pd.concat(
        rows,
        ignore_index=True,
    )
    out["experiment"] = (
        "delayed_physics"
    )
    return out


def joint_error_delay():
    rows = []

    for eps in ERRORS:
        for delay in DELAYS:
            case = [
                run_one(ReactivePolicy()),
                run_one(
                    VelocityHistoryPolicy(
                        window_k=3
                    )
                ),
                run_one(
                    CharacteristicWindowPolicy(
                        window_k=3,
                        amplitude_error=eps,
                        parameter_delay_steps=delay,
                    )
                ),
                run_one(
                    OracleWindowPolicy(
                        window_k=3
                    )
                ),
            ]

            df = add_reference_metrics(
                pd.DataFrame(case)
            )

            phys = df[
                df["policy"].str.startswith(
                    "physics_"
                )
            ].iloc[0]

            hist = df[
                df["policy"]
                == "history_velocity_k3"
            ].iloc[0]

            rows.append(
                {
                    "physics_error": eps,
                    "physics_delay": delay,
                    "physics_speedup":
                        phys["speedup_vs_reactive"],
                    "history_speedup":
                        hist["speedup_vs_reactive"],
                    "physics_over_history":
                        phys["speedup_vs_reactive"]
                        / hist["speedup_vs_reactive"],
                    "physics_traffic":
                        phys[
                            "traffic_overhead_vs_reactive"
                        ],
                    "history_traffic":
                        hist[
                            "traffic_overhead_vs_reactive"
                        ],
                }
            )

    return pd.DataFrame(rows)


def main():
    result_dir = ROOT / "results"
    plot_dir = ROOT / "plots"
    result_dir.mkdir(exist_ok=True)
    plot_dir.mkdir(exist_ok=True)

    eq = equal_budget()
    noisy = noisy_physics()
    delayed = delayed_physics()
    joint = joint_error_delay()

    eq.to_csv(
        result_dir / "rigor_equal_budget.csv",
        index=False,
    )
    noisy.to_csv(
        result_dir / "rigor_noisy_physics.csv",
        index=False,
    )
    delayed.to_csv(
        result_dir / "rigor_delayed_physics.csv",
        index=False,
    )
    joint.to_csv(
        result_dir / "rigor_joint_error_delay.csv",
        index=False,
    )

    # Noise plot
    for prefix in [
        "history_velocity_k3",
        "history_poly_k3",
        "physics_k3",
        "oracle_k3",
    ]:
        if prefix == "physics_k3":
            d = noisy[
                noisy["policy"].str.startswith(
                    "physics_k3"
                )
            ]
        else:
            d = noisy[
                noisy["policy"] == prefix
            ]

        plt.plot(
            100*d["physics_error"],
            d["speedup_vs_reactive"],
            marker="o",
            label=prefix,
        )

    plt.xlabel("Amplitude-model error (%)")
    plt.ylabel("Speedup vs reactive")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        plot_dir / "rigor_noisy_physics.png",
        dpi=180,
    )
    plt.close()

    # Delay plot
    phys = delayed[
        delayed["policy"].str.startswith(
            "physics_k3"
        )
    ]
    plt.plot(
        phys["physics_delay"],
        phys["speedup_vs_reactive"],
        marker="o",
        label="physics_k3",
    )

    hist = delayed[
        delayed["policy"]
        == "history_velocity_k3"
    ]
    plt.plot(
        hist["physics_delay"],
        hist["speedup_vs_reactive"],
        marker="o",
        label="history_velocity_k3",
    )

    oracle = delayed[
        delayed["policy"]
        == "oracle_k3"
    ]
    plt.plot(
        oracle["physics_delay"],
        oracle["speedup_vs_reactive"],
        marker="o",
        label="oracle_k3",
    )

    plt.xlabel("Physics-signal delay (timesteps)")
    plt.ylabel("Speedup vs reactive")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        plot_dir / "rigor_delayed_physics.png",
        dpi=180,
    )
    plt.close()

    # Joint heatmap
    pivot = joint.pivot(
        index="physics_error",
        columns="physics_delay",
        values="physics_over_history",
    )

    plt.imshow(
        pivot.values,
        origin="lower",
        aspect="auto",
    )
    plt.xticks(
        range(len(pivot.columns)),
        pivot.columns,
    )
    plt.yticks(
        range(len(pivot.index)),
        [f"{100*x:.0f}%" for x in pivot.index],
    )
    plt.xlabel("Physics-signal delay (timesteps)")
    plt.ylabel("Amplitude-model error")
    plt.colorbar(
        label="Physics speedup / history speedup"
    )
    plt.tight_layout()
    plt.savefig(
        plot_dir / "rigor_error_delay_heatmap.png",
        dpi=180,
    )
    plt.close()

    print("\n=== EQUAL BUDGET ===")
    print(
        eq[
            [
                "policy",
                "total_cycles",
                "bytes_transferred",
                "traffic_overhead_vs_reactive",
                "speedup_vs_reactive",
            ]
        ].to_string(index=False)
    )

    print("\n=== NOISY PHYSICS ===")
    print(
        noisy[
            noisy["policy"].isin(
                [
                    "history_velocity_k3",
                    "history_poly_k3",
                    "oracle_k3",
                ]
            )
            | noisy["policy"].str.startswith(
                "physics_k3"
            )
        ][
            [
                "physics_error",
                "policy",
                "speedup_vs_reactive",
                "traffic_overhead_vs_reactive",
            ]
        ].to_string(index=False)
    )

    print("\n=== DELAYED PHYSICS ===")
    print(
        delayed[
            delayed["policy"].isin(
                [
                    "history_velocity_k3",
                    "history_poly_k3",
                    "oracle_k3",
                ]
            )
            | delayed["policy"].str.startswith(
                "physics_k3"
            )
        ][
            [
                "physics_delay",
                "policy",
                "speedup_vs_reactive",
                "traffic_overhead_vs_reactive",
            ]
        ].to_string(index=False)
    )

    print("\n=== ERROR × DELAY REGIME MAP ===")
    print(
        joint.to_string(index=False)
    )


if __name__ == "__main__":
    main()
