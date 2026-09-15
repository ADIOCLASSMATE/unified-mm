"""Select an explicit whole-image experiment for the shared launch/acceptance tools."""


def joint_experiment_protocol(name):
    if name == "z":
        from utils import joint_dit_protocol as protocol
    elif name == "z-b":
        from utils import joint_b_protocol as protocol
    else:
        raise ValueError(f"Unknown joint-image experiment: {name!r}")
    return protocol
