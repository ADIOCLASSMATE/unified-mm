"""Spatial token orders shared by generation and understanding evaluation."""
from functools import lru_cache


@lru_cache(maxsize=4096)
def halton_image_positions(image_tokens: int, side: int, shift=(0.0, 0.0)) -> tuple[int, ...]:
    """Return physical positions in reveal order, using bases 2 and 3."""
    if side <= 0 or image_tokens != side * side:
        raise ValueError("Halton image order requires a square token grid")
    if len(shift) != 2 or not all(0.0 <= value < 1.0 for value in shift):
        raise ValueError("Halton shifts must be two coordinates in [0, 1)")

    def halton(index, base):
        value, scale = 0.0, 1.0 / float(base)
        while index > 0:
            value += (index % base) * scale
            index //= base
            scale /= float(base)
        return value

    seen, order, index = set(), [], 1
    while len(order) < image_tokens and index < image_tokens * 32:
        row = min(side - 1, int(((halton(index, 2) + shift[0]) % 1.0) * side))
        col = min(side - 1, int(((halton(index, 3) + shift[1]) % 1.0) * side))
        position = row * side + col
        if position not in seen:
            seen.add(position)
            order.append(position)
        index += 1
    order.extend(position for position in range(image_tokens) if position not in seen)
    return tuple(order)
