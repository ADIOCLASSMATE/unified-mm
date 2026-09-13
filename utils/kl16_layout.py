"""Spatial contract shared by offline KL16 cache tools."""


def kl16_layout(image_size: int) -> tuple[int, int]:
    image_size = int(image_size)
    if image_size < 16 or image_size % 16:
        raise ValueError("KL16 image_size must be a positive multiple of 16")
    side = image_size // 16
    return side, side * side
