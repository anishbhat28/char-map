# char-map

Minimal falsification testbed for **characteristic-aware hardware prefetching**.

## Hypothesis
If governing physics tells us where information will propagate, can hardware use that information to prefetch data before it is requested, reducing stalls relative to reactive and history-only policies?

Initial physics: 1-D periodic constant advection, `u_t + c u_x = 0`, with backward characteristic `chi_c(x,t) = (x - c t) mod L`.

## Included
- ring accelerator model
- finite per-PE local storage + LRU eviction
- asynchronous transfers with per-hop latency
- constant-advection information-flow trace generator
- policies: reactive, last-value history, velocity-history, characteristic, oracle
- baseline experiment + smoke tests

## Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
python main.py --config configs/default.yaml
python tests/test_smoke.py
```

The meaningful research comparison is **characteristic vs history_velocity**, not merely characteristic vs reactive. Oracle gives an upper bound on available opportunity.

## Important modeling choice
The simulator models a stream of output queries at increasing query times against a fixed initial-condition field, matching the characteristic-aligned neural-operator setting. At query timestep `t`, destination cell `i` requests the initial-condition block at the backward characteristic origin corresponding to that query time. This avoids incorrectly caching a stale one-step stencil value forever.

This is a **falsification simulator**, not yet a cycle-accurate accelerator model. If the characteristic policy does not beat sensible history-based prediction across broad parameter regimes here, do not proceed to a more detailed hardware implementation.
