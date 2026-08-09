from common import *
import matplotlib.pyplot as plt

def main():
    rows=[]
    for l in [1,2,4,8,16,32]:
        d=run_family({'hop_latency':l}); d['hop_latency']=l; d['transfer_compute_ratio']=l/float(BASE['hardware']['compute_latency']); rows.append(d)
    out=pd.concat(rows,ignore_index=True); out.to_csv(ROOT/'results/sweep_latency.csv',index=False)
    for p in out.policy.unique():
        d=out[out.policy==p]; plt.plot(d.hop_latency,d.speedup_vs_reactive,marker='o',label=p)
    plt.xlabel('Per-hop latency (cycles)'); plt.ylabel('Speedup vs reactive'); plt.legend(); plt.tight_layout(); plt.savefig(ROOT/'plots/sweep_latency.png',dpi=180); plt.close()
if __name__=='__main__': main()
