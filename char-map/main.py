import argparse
from pathlib import Path
import pandas as pd
from src.config import load_yaml,sim_config_from_dict
from src.simulator import Simulator
from src.policies import ReactivePolicy,LastValuePolicy,VelocityHistoryPolicy,CharacteristicPolicy,OraclePolicy
from src.metrics import add_speedups

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='configs/default.yaml'); ap.add_argument('--save',default='results/baselines.csv'); args=ap.parse_args()
    raw=load_yaml(args.config); cfg=sim_config_from_dict(raw); eps=float(raw['policy'].get('velocity_error',0.0))
    policies=[ReactivePolicy(),LastValuePolicy(),VelocityHistoryPolicy(),CharacteristicPolicy(eps),OraclePolicy()]
    sums=[]; frames=[]
    for p in policies:
        d,s=Simulator(cfg,p).run(); sums.append(s); frames.append(d)
    out=add_speedups(pd.DataFrame(sums)); Path(args.save).parent.mkdir(parents=True,exist_ok=True); out.to_csv(args.save,index=False)
    pd.concat(frames,ignore_index=True).to_csv(Path(args.save).with_name(Path(args.save).stem+'_accesses.csv'),index=False)
    print(out[['policy','total_cycles','stall_cycles','hit_rate','prefetch_hit_rate','bytes_transferred','speedup_vs_reactive']].to_string(index=False))
if __name__=='__main__': main()
