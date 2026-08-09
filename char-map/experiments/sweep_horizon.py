from common import *
import matplotlib.pyplot as plt

def main():
    rows=[]
    for h in [0,1,2,4,8,16]:
        d=run_family({'horizon':h}); d['horizon']=h; rows.append(d)
    out=pd.concat(rows,ignore_index=True); out.to_csv(ROOT/'results/sweep_horizon.csv',index=False)
    for p in out.policy.unique():
        d=out[out.policy==p]; plt.plot(d.horizon,d.speedup_vs_reactive,marker='o',label=p)
    plt.xlabel('Prediction horizon (timesteps)'); plt.ylabel('Speedup vs reactive'); plt.legend(); plt.tight_layout(); plt.savefig(ROOT/'plots/sweep_horizon.png',dpi=180); plt.close()
if __name__=='__main__': main()
