"""Deterministic coverage of the evaluation ordering without consuming RNG."""


def evenly_spaced_image_indices(samples: int, count: int) -> list[int]:
    if samples <= 0 or count < 0:
        raise ValueError("samples must be positive and image count nonnegative")
    count = min(samples, count)
    if count <= 1:
        return [0] if count else []
    return [i * (samples - 1) // (count - 1) for i in range(count)]
