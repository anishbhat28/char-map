from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.simulator import Simulator,SimConfig
from src.policies import ReactivePolicy,CharacteristicPolicy,OraclePolicy

def cfg():
    return SimConfig(16,16,20,2,1.0,1.0,1/16,4,4,2,64,1)

def run_tests():
    sim=Simulator(cfg(),ReactivePolicy()); assert sim.hardware.ring_distance(0,15)==1; assert sim.physics.source_index(0,1)==15
    _,r=Simulator(cfg(),ReactivePolicy()).run(); _,c=Simulator(cfg(),CharacteristicPolicy()).run(); _,o=Simulator(cfg(),OraclePolicy()).run()
    assert c['total_cycles']<=r['total_cycles']; assert o['total_cycles']<=r['total_cycles']
    print('All smoke tests passed.')
if __name__=='__main__': run_tests()
