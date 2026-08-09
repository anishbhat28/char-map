import pandas as pd

def summarize_accesses(df, *, total_cycles, total_bytes, total_hops, prefetch_bytes, evictions):
    return {
        'requests': int(len(df)),
        'stall_cycles': int(df['stall_cycles'].sum()) if len(df) else 0,
        'mean_stall': float(df['stall_cycles'].mean()) if len(df) else 0.0,
        'hit_rate': float(df['hit'].mean()) if len(df) else 0.0,
        'prefetch_hit_rate': float(df['prefetched_hit'].mean()) if len(df) else 0.0,
        'total_cycles': int(total_cycles),
        'bytes_transferred': int(total_bytes),
        'prefetch_bytes': int(prefetch_bytes),
        'transfer_hops': int(total_hops),
        'evictions': int(evictions),
    }

def add_speedups(df):
    out=df.copy()
    base=float(out.loc[out.policy=='reactive','total_cycles'].iloc[0])
    out['speedup_vs_reactive']=base/out['total_cycles']
    if 'oracle' in set(out.policy):
        so=float(out.loc[out.policy=='oracle','speedup_vs_reactive'].iloc[0])
        den=so-1.0
        out['oracle_capture']=((out['speedup_vs_reactive']-1.0)/den) if abs(den)>1e-12 else 0.0
    return out
