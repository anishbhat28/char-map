from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import pandas as pd
from src.config import load_yaml,sim_config_from_dict
from src.simulator import Simulator
from src.policies import ReactivePolicy,LastValuePolicy,VelocityHistoryPolicy,CharacteristicPolicy,OraclePolicy
from src.metrics import add_speedups
BASE=load_yaml(ROOT/'configs/default.yaml')
def run_family(overrides=None,velocity_error=0.0):
    cfg=sim_config_from_dict(BASE,overrides or {})
    rows=[]
    for p in [ReactivePolicy(),LastValuePolicy(),VelocityHistoryPolicy(),CharacteristicPolicy(velocity_error),OraclePolicy()]:
        _,s=Simulator(cfg,p).run(); rows.append(s)
    return add_speedups(pd.DataFrame(rows))
