from dataclasses import dataclass
import pandas as pd
from .hardware import RingHardware
from .physics import ConstantAdvection
from .metrics import summarize_accesses

@dataclass
class SimConfig:
    num_pes:int; num_cells:int; timesteps:int; warmup_steps:int
    domain_length:float; velocity:float; dt:float
    hop_latency:int; compute_latency:int; local_capacity:int; block_size_bytes:int; horizon:int

class Simulator:
    def __init__(self,cfg,policy):
        self.cfg=cfg; self.policy=policy
        self.physics=ConstantAdvection(cfg.velocity,cfg.domain_length,cfg.num_cells,cfg.dt,cfg.num_pes)
        self.hardware=RingHardware(cfg.num_pes,cfg.hop_latency,cfg.compute_latency,cfg.local_capacity,cfg.block_size_bytes)
        self.hardware.initialize_home_blocks(cfg.num_cells)
        self.policy.reset(); self.rows=[]; self.cycle=0

    def run(self):
        for t in range(self.cfg.timesteps):
            self.hardware.advance(self.cycle)
            preds=self.policy.predict(physics=self.physics,timestep=t,cycle=self.cycle,horizon=self.cfg.horizon,compute_latency=self.cfg.compute_latency)
            for p in preds:
                self.hardware.schedule_prefetch(p.block_id,self.physics.owner_pe(p.block_id),p.target_pe,self.cycle)
            reqs=self.physics.requests_at(t,self.cycle)
            results=[]; stalls=[]
            for r in reqs:
                res=self.hardware.serve(r,self.cycle)
                results.append((r,res)); stalls.append(res.stall_cycles)
            step_stall=max(stalls) if stalls else 0
            if t>=self.cfg.warmup_steps:
                for r,res in results:
                    self.rows.append({'policy':self.policy.name,'timestep':t,'cycle':self.cycle,'consumer_pe':r.consumer_pe,'block_id':r.block_id,'producer_pe':r.producer_pe,'hit':res.hit,'stall_cycles':res.stall_cycles,'step_stall_cycles':step_stall,'transfer_hops':res.transfer_hops,'prefetched_hit':res.prefetched_hit})
            self.policy.observe_requests(reqs)
            self.cycle += step_stall + self.cfg.compute_latency
        df=pd.DataFrame(self.rows)
        s=summarize_accesses(df,total_cycles=self.cycle,total_bytes=self.hardware.total_bytes,total_hops=self.hardware.total_hops,prefetch_bytes=self.hardware.total_prefetch_bytes,evictions=self.hardware.evictions)
        s['policy']=self.policy.name
        return df,s
