from common import *
import matplotlib.pyplot as plt

def main():
    rows=[]
    for c in [1,2,4,8,16,32]:
        d=run_family({'local_capacity':c}); d['local_capacity']=c; rows.append(d)
    out=pd.concat(rows,ignore_index=True); out.to_csv(ROOT/'results/sweep_memory.csv',index=False)
    for p in out.policy.unique():
        d=out[out.policy==p]; plt.plot(d.local_capacity,d.speedup_vs_reactive,marker='o',label=p)
    plt.xlabel('Local capacity (blocks/PE)'); plt.ylabel('Speedup vs reactive'); plt.legend(); plt.tight_layout(); plt.savefig(ROOT/'plots/sweep_memory.png',dpi=180); plt.close()
if __name__=='__main__': main()
