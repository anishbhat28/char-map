from common import *
import matplotlib.pyplot as plt

def main():
    rows=[]
    for l in [1,2,4,8,16]:
        for c in [1,2,4,8,16]:
            d=run_family({'hop_latency':l,'local_capacity':c})
            sc=float(d.loc[d.policy=='characteristic','speedup_vs_reactive'].iloc[0])
            sh=float(d.loc[d.policy=='history_velocity','speedup_vs_reactive'].iloc[0])
            rows.append({'hop_latency':l,'local_capacity':c,'char_advantage_over_history':sc/sh if sh else float('nan')})
    out=pd.DataFrame(rows); out.to_csv(ROOT/'results/regime_heatmap.csv',index=False)
    piv=out.pivot(index='local_capacity',columns='hop_latency',values='char_advantage_over_history')
    plt.imshow(piv.values,aspect='auto',origin='lower'); plt.xticks(range(len(piv.columns)),piv.columns); plt.yticks(range(len(piv.index)),piv.index)
    plt.xlabel('Per-hop latency'); plt.ylabel('Local capacity'); plt.colorbar(label='Characteristic speedup / history speedup'); plt.tight_layout(); plt.savefig(ROOT/'plots/regime_heatmap.png',dpi=180); plt.close()
if __name__=='__main__': main()
