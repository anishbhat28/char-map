Copy the files into the matching locations in your existing repo, replacing physics.py and policies.py.

Then run:

python tests/test_smoke.py
python main.py --config configs/default.yaml
python experiments/variable_advection.py
python experiments/sweep_physics_error.py
