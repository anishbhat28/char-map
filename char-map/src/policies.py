from collections import defaultdict, deque
from .types import Prediction

class Policy:
    name="base"
    def reset(self): pass
    def observe_requests(self, requests): pass
    def predict(self, **kwargs): return []

class ReactivePolicy(Policy): name="reactive"

class LastValuePolicy(Policy):
    name="history_last"
    def __init__(self): self.last_block={}
    def reset(self): self.last_block={}
    def observe_requests(self,requests):
        for r in requests:self.last_block[r.consumer_pe]=r.block_id
    def predict(self,*,physics,timestep,cycle,horizon,compute_latency):
        return [Prediction(b,pe,cycle+horizon*compute_latency,1.0) for pe,b in self.last_block.items()]

class VelocityHistoryPolicy(Policy):
    name="history_velocity"
    def __init__(self): self.history=defaultdict(lambda:deque(maxlen=2))
    def reset(self): self.history=defaultdict(lambda:deque(maxlen=2))
    def observe_requests(self,requests):
        for r in requests:self.history[r.consumer_pe].append(r.block_id)
    @staticmethod
    def cyclic_delta(a,b,N):
        return min([b-a,b-a+N,b-a-N],key=abs)
    def predict(self,*,physics,timestep,cycle,horizon,compute_latency):
        out=[]
        for pe in range(physics.num_pes):
            h=self.history[pe]
            if not h: continue
            pred=h[-1] if len(h)==1 else (h[-1]+(horizon+1)*self.cyclic_delta(h[-2],h[-1],physics.N))%physics.N
            out.append(Prediction(int(pred),pe,cycle+horizon*compute_latency,1.0))
        return out

class CharacteristicPolicy(Policy):
    name="characteristic"
    def __init__(self,velocity_error=0.0,amplitude_error=0.0):
        self.velocity_error=velocity_error; self.amplitude_error=amplitude_error
    def predict(self,*,physics,timestep,cycle,horizon,compute_latency):
        c_est=physics.c*(1+self.velocity_error); amp_scale=1+self.amplitude_error
        return [Prediction(physics.source_index(pe%physics.N,horizon=timestep+horizon+1,
                    velocity_override=c_est,amplitude_scale=amp_scale),
                    pe,cycle+horizon*compute_latency,1.0) for pe in range(physics.num_pes)]

class OraclePolicy(Policy):
    name="oracle"
    def predict(self,*,physics,timestep,cycle,horizon,compute_latency):
        return [Prediction(physics.source_index(pe%physics.N,horizon=timestep+horizon+1,amplitude_scale=1.0),
                    pe,cycle+horizon*compute_latency,1.0) for pe in range(physics.num_pes)]
