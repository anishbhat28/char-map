from common import *
import matplotlib.pyplot as plt

def main():
    rows=[]
    for e in [0.0,0.01,0.02,0.05,0.1,0.2,0.5]:
        d=run_family({},e); d['velocity_error']=e; rows.append(d)
    out=pd.concat(rows,ignore_index=True); out.to_csv(ROOT/'results/sweep_noise.csv',index=False)
    for p in ['history_velocity','characteristic','oracle']:
        d=out[out.policy==p]; plt.plot(d.velocity_error,d.speedup_vs_reactive,marker='o',label=p)
    plt.xlabel('Relative velocity error'); plt.ylabel('Speedup vs reactive'); plt.legend(); plt.tight_layout(); plt.savefig(ROOT/'plots/sweep_noise.png',dpi=180); plt.close()
if __name__=='__main__': main()
