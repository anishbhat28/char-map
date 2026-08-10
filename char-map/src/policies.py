from collections import defaultdict, deque
import numpy as np
from .types import Prediction

class Policy:
    name="base"
    def reset(self): pass
    def observe_requests(self, requests): pass
    def predict(self, **kwargs): return []

class ReactivePolicy(Policy):
    name="reactive"

class LastValuePolicy(Policy):
    name="history_last"
    def __init__(self): self.last_block={}
    def reset(self): self.last_block={}
    def observe_requests(self,requests):
        for r in requests:
            self.last_block[r.consumer_pe]=r.block_id
    def predict(self,*,physics,timestep,cycle,horizon,compute_latency):
        return [Prediction(b,pe,cycle+horizon*compute_latency,1.0)
                for pe,b in self.last_block.items()]

class VelocityHistoryPolicy(Policy):
    name="history_velocity"
    def __init__(self, window_k=1):
        self.history=defaultdict(lambda:deque(maxlen=2))
        self.window_k=window_k
    def reset(self):
        self.history=defaultdict(lambda:deque(maxlen=2))
    def observe_requests(self,requests):
        for r in requests:
            self.history[r.consumer_pe].append(r.block_id)
    @staticmethod
    def cyclic_delta(a,b,N):
        return min([b-a,b-a+N,b-a-N],key=abs)
    def predict(self,*,physics,timestep,cycle,horizon,compute_latency):
        preds=[]
        radius=max(0,self.window_k//2)
        for pe in range(physics.num_pes):
            h=self.history[pe]
            if not h: continue
            if len(h)==1:
                center=h[-1]
            else:
                delta=self.cyclic_delta(h[-2],h[-1],physics.N)
                center=(h[-1]+(horizon+1)*delta)%physics.N
            for off in range(-radius,radius+1):
                preds.append(Prediction(int((center+off)%physics.N),pe,
                                        cycle+horizon*compute_latency,1.0/(1+abs(off))))
        return preds

class PolynomialHistoryPolicy(Policy):
    """
    Fits a low-degree polynomial to unwrapped recent source indices for each PE,
    extrapolates to t+horizon, then prefetches a symmetric K-block window.
    """
    name="history_poly"
    def __init__(self,history_len=6,degree=2,window_k=3):
        self.history_len=history_len
        self.degree=degree
        self.window_k=window_k
        self.history=defaultdict(lambda:deque(maxlen=history_len))
    def reset(self):
        self.history=defaultdict(lambda:deque(maxlen=self.history_len))
    def observe_requests(self,requests):
        for r in requests:
            self.history[r.consumer_pe].append(r.block_id)
    @staticmethod
    def unwrap(vals,N):
        if not vals: return []
        out=[float(vals[0])]
        for v in vals[1:]:
            prev=out[-1]
            candidates=[float(v)+k*N for k in (-2,-1,0,1,2)]
            out.append(min(candidates,key=lambda x:abs(x-prev)))
        return out
    def predict(self,*,physics,timestep,cycle,horizon,compute_latency):
        preds=[]
        radius=self.window_k//2
        for pe in range(physics.num_pes):
            h=list(self.history[pe])
            if not h: continue
            if len(h)<3:
                center=h[-1]
            else:
                y=np.array(self.unwrap(h,physics.N),dtype=float)
                x=np.arange(len(y),dtype=float)
                deg=min(self.degree,len(y)-1)
                coeff=np.polyfit(x,y,deg)
                # latest observed corresponds to current timestep-1;
                # predict horizon+1 intervals beyond latest observation
                xq=(len(y)-1)+(horizon+1)
                center=int(round(np.polyval(coeff,xq)))%physics.N
            for off in range(-radius,radius+1):
                preds.append(Prediction((center+off)%physics.N,pe,
                                        cycle+horizon*compute_latency,1.0/(1+abs(off))))
        return preds

class MarkovHistoryPolicy(Policy):
    """
    Online order-2 transition table:
        (b[t-2], b[t-1]) -> most frequent next delta/state.
    Falls back to velocity extrapolation if a state has not been seen.
    Prefetches a K-block window around the predicted center.
    """
    name="history_markov"
    def __init__(self,window_k=3):
        self.window_k=window_k
        self.hist=defaultdict(lambda:deque(maxlen=3))
        self.table=defaultdict(lambda:defaultdict(int))
    def reset(self):
        self.hist=defaultdict(lambda:deque(maxlen=3))
        self.table=defaultdict(lambda:defaultdict(int))
    @staticmethod
    def cyclic_delta(a,b,N):
        return min([b-a,b-a+N,b-a-N],key=abs)
    def observe_requests(self,requests):
        for r in requests:
            pe=r.consumer_pe
            h=self.hist[pe]
            if len(h)>=2:
                state=(h[-2],h[-1])
                self.table[(pe,state)][r.block_id]+=1
            h.append(r.block_id)
    def predict(self,*,physics,timestep,cycle,horizon,compute_latency):
        preds=[]
        radius=self.window_k//2
        for pe in range(physics.num_pes):
            h=self.hist[pe]
            if not h: continue
            center=h[-1]
            # iterative multi-step prediction
            seq=list(h)
            steps=horizon+1
            for _ in range(steps):
                if len(seq)>=2:
                    state=(seq[-2],seq[-1])
                    counts=self.table.get((pe,state),{})
                    if counts:
                        nxt=max(counts.items(),key=lambda kv:kv[1])[0]
                    else:
                        delta=self.cyclic_delta(seq[-2],seq[-1],physics.N)
                        nxt=(seq[-1]+delta)%physics.N
                else:
                    nxt=seq[-1]
                seq.append(nxt)
            center=seq[-1]
            for off in range(-radius,radius+1):
                preds.append(Prediction((center+off)%physics.N,pe,
                                        cycle+horizon*compute_latency,1.0/(1+abs(off))))
        return preds

class CharacteristicPolicy(Policy):
    name="characteristic"
    def __init__(self,velocity_error=0.0,amplitude_error=0.0):
        self.velocity_error=velocity_error
        self.amplitude_error=amplitude_error
    def predict(self,*,physics,timestep,cycle,horizon,compute_latency):
        c_est=physics.c*(1+self.velocity_error)
        amp_scale=1+self.amplitude_error
        qstep=timestep+horizon+1
        return [Prediction(physics.source_index(pe%physics.N,horizon=qstep,
                        velocity_override=c_est,amplitude_scale=amp_scale),
                        pe,cycle+horizon*compute_latency,1.0)
                for pe in range(physics.num_pes)]

class UncertaintyCharacteristicPolicy(Policy):
    name="characteristic_window"
    def __init__(self,window_k=3,velocity_error=0.0,amplitude_error=0.0):
        if window_k<1 or window_k%2==0:
            raise ValueError("window_k must be positive odd")
        self.window_k=window_k
        self.velocity_error=velocity_error
        self.amplitude_error=amplitude_error
    def predict(self,*,physics,timestep,cycle,horizon,compute_latency):
        c_est=physics.c*(1+self.velocity_error)
        amp_scale=1+self.amplitude_error
        qstep=timestep+horizon+1
        radius=self.window_k//2
        out=[]
        for pe in range(physics.num_pes):
            center=physics.source_index(pe%physics.N,horizon=qstep,
                                        velocity_override=c_est,amplitude_scale=amp_scale)
            for off in range(-radius,radius+1):
                out.append(Prediction((center+off)%physics.N,pe,
                                      cycle+horizon*compute_latency,1.0/(1+abs(off))))
        return out

class OraclePolicy(Policy):
    name="oracle"
    def predict(self,*,physics,timestep,cycle,horizon,compute_latency):
        qstep=timestep+horizon+1
        return [Prediction(physics.source_index(pe%physics.N,horizon=qstep,amplitude_scale=1.0),
                           pe,cycle+horizon*compute_latency,1.0)
                for pe in range(physics.num_pes)]
