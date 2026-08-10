from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import pandas as pd
import matplotlib.pyplot as plt
from src.config import load_yaml,sim_config_from_dict
from src.simulator import Simulator
from src.policies import ReactivePolicy,VelocityHistoryPolicy,CharacteristicPolicy,OraclePolicy
from src.metrics import add_speedups

BASE=load_yaml(ROOT/"configs/default.yaml")
ERRORS=[0,.01,.02,.05,.10,.20,.50]

def main():
    rows=[]
    for eps in ERRORS:
        cfg=sim_config_from_dict(BASE,overrides={"physics_model":"variable","variable_amplitude":.75,
                                                "variable_omega":4.0,"rk4_substeps_per_dt":8})
        case=[]
        for p in [ReactivePolicy(),VelocityHistoryPolicy(),
                  CharacteristicPolicy(amplitude_error=eps),OraclePolicy()]:
            _,s=Simulator(cfg,p).run(); case.append(s)
        df=add_speedups(pd.DataFrame(case)); df["physics_error"]=eps; rows.append(df)
    out=pd.concat(rows,ignore_index=True)
    (ROOT/"results").mkdir(exist_ok=True); (ROOT/"plots").mkdir(exist_ok=True)
    out.to_csv(ROOT/"results/sweep_physics_error.csv",index=False)
    for p in ["history_velocity","characteristic","oracle"]:
        d=out[out.policy==p]
        plt.plot(100*d.physics_error,d.speedup_vs_reactive,marker="o",label=p)
    plt.xlabel("Amplitude-model error (%)"); plt.ylabel("Speedup vs reactive"); plt.legend(); plt.tight_layout()
    plt.savefig(ROOT/"plots/sweep_physics_error.png",dpi=180); plt.close()
    print(out[["physics_error","policy","total_cycles","hit_rate","speedup_vs_reactive","oracle_capture"]].to_string(index=False))
if __name__=="__main__": main()
