import yaml

from .simulator import SimConfig


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def sim_config_from_dict(
    cfg,
    overrides=None,
):

    overrides = (
        overrides or {}
    )

    s = cfg["simulation"]
    p = cfg["physics"]
    h = cfg["hardware"]
    pol = cfg["policy"]

    d = dict(
        num_pes=s["num_pes"],
        num_cells=s["num_cells"],
        timesteps=s["timesteps"],
        warmup_steps=s.get(
            "warmup_steps",
            0,
        ),

        domain_length=
            p["domain_length"],

        velocity=
            p["velocity"],

        dt=
            p["dt"],

        physics_model=
            p.get(
                "model",
                "constant",
            ),

        variable_amplitude=
            p.get(
                "variable_amplitude",
                0.0,
            ),

        variable_omega=
            p.get(
                "variable_omega",
                0.0,
            ),

        rk4_substeps_per_dt=
            p.get(
                "rk4_substeps_per_dt",
                8,
            ),

        hop_latency=
            h["hop_latency"],

        compute_latency=
            h["compute_latency"],

        local_capacity=
            h["local_capacity"],

        block_size_bytes=
            h["block_size_bytes"],

        horizon=
            pol["horizon"],
    )

    d.update(
        overrides
    )

    return SimConfig(**d)
