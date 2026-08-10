from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import pandas as pd
import matplotlib.pyplot as plt

from src.config import load_yaml,sim_config_from_dict
from src.simulator import Simulator
from src.policies import (
    ReactivePolicy,
    VelocityHistoryPolicy,
    CharacteristicPolicy,
    UncertaintyCharacteristicPolicy,
    OraclePolicy,
)
from src.metrics import add_speedups

BASE=load_yaml(ROOT/"configs/default.yaml")

ERRORS=[0.00,0.01,0.02,0.05,0.10,0.20]
WINDOWS=[1,3,5,7]

TRUE_AMPLITUDE=0.75
OMEGA=4.0

def run_policy(cfg,policy):
    _,summary=Simulator(cfg,policy).run()
    return summary

def main():
    rows=[]

    for eps in ERRORS:
        cfg=sim_config_from_dict(
            BASE,
            overrides={
                "physics_model":"variable",
                "variable_amplitude":TRUE_AMPLITUDE,
                "variable_omega":OMEGA,
                "rk4_substeps_per_dt":8,
            },
        )

        # Common references for this error level
        base_rows=[]
        for p in [
            ReactivePolicy(),
            VelocityHistoryPolicy(),
            CharacteristicPolicy(amplitude_error=eps),
            OraclePolicy(),
        ]:
            s=run_policy(cfg,p)
            s["physics_error"]=eps
            s["window_k"]=1 if p.name=="characteristic" else 0
            base_rows.append(s)

        base_df=add_speedups(pd.DataFrame(base_rows))
        rows.append(base_df)

        # Windowed policies
        reactive_cycles=float(
            base_df.loc[base_df.policy=="reactive","total_cycles"].iloc[0]
        )
        oracle_speed=float(
            base_df.loc[base_df.policy=="oracle","speedup_vs_reactive"].iloc[0]
        )
        reactive_bytes=float(
            base_df.loc[base_df.policy=="reactive","bytes_transferred"].iloc[0]
        )

        for k in WINDOWS:
            p=UncertaintyCharacteristicPolicy(
                window_k=k,
                amplitude_error=eps,
            )
            s=run_policy(cfg,p)
            s["physics_error"]=eps
            s["window_k"]=k
            s["speedup_vs_reactive"]=reactive_cycles/float(s["total_cycles"])
            denom=oracle_speed-1.0
            s["oracle_capture"]=(
                (s["speedup_vs_reactive"]-1.0)/denom if abs(denom)>1e-12 else 0.0
            )
            s["traffic_overhead_vs_reactive"]=(
                float(s["bytes_transferred"])/reactive_bytes if reactive_bytes else float("nan")
            )
            rows.append(pd.DataFrame([s]))

    out=pd.concat(rows,ignore_index=True)

    # Ensure traffic overhead is present for all rows.
    for eps in ERRORS:
        mask=out.physics_error==eps
        reactive_bytes=float(out.loc[mask & (out.policy=="reactive"),"bytes_transferred"].iloc[0])
        out.loc[mask,"traffic_overhead_vs_reactive"]=(
            out.loc[mask,"bytes_transferred"]/reactive_bytes
        )

    (ROOT/"results").mkdir(exist_ok=True)
    (ROOT/"plots").mkdir(exist_ok=True)

    out.to_csv(
        ROOT/"results/sweep_uncertainty_window.csv",
        index=False,
    )

    # Plot 1: speedup vs error for each K
    for k in WINDOWS:
        d=out[(out.policy=="characteristic_window") & (out.window_k==k)]
        plt.plot(
            100*d.physics_error,
            d.speedup_vs_reactive,
            marker="o",
            label=f"K={k}",
        )
    hist=out[out.policy=="history_velocity"].drop_duplicates("physics_error")
    oracle=out[out.policy=="oracle"].drop_duplicates("physics_error")
    plt.plot(100*hist.physics_error,hist.speedup_vs_reactive,marker="o",label="history_velocity")
    plt.plot(100*oracle.physics_error,oracle.speedup_vs_reactive,marker="o",label="oracle")
    plt.xlabel("Amplitude-model error (%)")
    plt.ylabel("Speedup vs reactive")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ROOT/"plots/uncertainty_speedup.png",dpi=180)
    plt.close()

    # Plot 2: traffic overhead vs error
    for k in WINDOWS:
        d=out[(out.policy=="characteristic_window") & (out.window_k==k)]
        plt.plot(
            100*d.physics_error,
            d.traffic_overhead_vs_reactive,
            marker="o",
            label=f"K={k}",
        )
    plt.xlabel("Amplitude-model error (%)")
    plt.ylabel("Traffic overhead vs reactive")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ROOT/"plots/uncertainty_traffic.png",dpi=180)
    plt.close()

    # Compact printed table: only history, oracle, and window policies.
    view=out[
        out.policy.isin(["history_velocity","characteristic_window","oracle"])
    ][[
        "physics_error",
        "policy",
        "window_k",
        "total_cycles",
        "hit_rate",
        "bytes_transferred",
        "traffic_overhead_vs_reactive",
        "speedup_vs_reactive",
        "oracle_capture",
    ]].sort_values(["physics_error","policy","window_k"])

    print(view.to_string(index=False))

if __name__=="__main__":
    main()
