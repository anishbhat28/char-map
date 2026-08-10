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
    PolynomialHistoryPolicy,
    MarkovHistoryPolicy,
    UncertaintyCharacteristicPolicy,
    OraclePolicy,
)
from src.metrics import add_speedups

BASE=load_yaml(ROOT/"configs/default.yaml")
ERRORS=[0.00,0.02,0.05,0.10]

def run_standard_variable():
    rows=[]
    for eps in ERRORS:
        cfg=sim_config_from_dict(
            BASE,
            overrides={
                "physics_model":"variable",
                "variable_amplitude":0.75,
                "variable_omega":4.0,
                "rk4_substeps_per_dt":8,
            },
        )
        policies=[
            ReactivePolicy(),
            VelocityHistoryPolicy(window_k=3),
            PolynomialHistoryPolicy(history_len=6,degree=2,window_k=3),
            MarkovHistoryPolicy(window_k=3),
            UncertaintyCharacteristicPolicy(window_k=3,amplitude_error=eps),
            OraclePolicy(),
        ]
        case=[]
        for p in policies:
            _,s=Simulator(cfg,p).run()
            s["physics_error"]=eps
            case.append(s)
        rows.append(add_speedups(pd.DataFrame(case)))
    return pd.concat(rows,ignore_index=True)

def main():
    out=run_standard_variable()
    (ROOT/"results").mkdir(exist_ok=True)
    (ROOT/"plots").mkdir(exist_ok=True)
    out.to_csv(ROOT/"results/strong_baselines.csv",index=False)

    # Traffic normalized per error against reactive
    pieces=[]
    for eps in ERRORS:
        d=out[out.physics_error==eps].copy()
        rbytes=float(d.loc[d.policy=="reactive","bytes_transferred"].iloc[0])
        d["traffic_overhead_vs_reactive"]=d["bytes_transferred"]/rbytes
        pieces.append(d)
    out=pd.concat(pieces,ignore_index=True)
    out.to_csv(ROOT/"results/strong_baselines_with_traffic.csv",index=False)

    for p in ["history_velocity","history_poly","history_markov","characteristic_window","oracle"]:
        d=out[out.policy==p]
        plt.plot(100*d.physics_error,d.speedup_vs_reactive,marker="o",label=p)
    plt.xlabel("Physics amplitude error (%)")
    plt.ylabel("Speedup vs reactive")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ROOT/"plots/strong_baselines_speedup.png",dpi=180)
    plt.close()

    view=out[out.policy.isin(
        ["history_velocity","history_poly","history_markov","characteristic_window","oracle"]
    )][[
        "physics_error","policy","total_cycles","hit_rate","bytes_transferred",
        "traffic_overhead_vs_reactive","speedup_vs_reactive","oracle_capture"
    ]]
    print(view.to_string(index=False))

if __name__=="__main__":
    main()
