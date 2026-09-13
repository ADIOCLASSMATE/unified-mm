import io

from PIL import Image, ImageDraw

from utils.image_near_duplicates import NearDuplicateIndex, perceptual_hashes, write_index


def test_perceptual_hash_is_stable_under_resizing_and_jpeg_reencoding(tmp_path):
    im = Image.new("RGB", (800, 600), "white")
    draw = ImageDraw.Draw(im)
    draw.rectangle((15, 30, 340, 230), fill="black")
    draw.ellipse((480, 190, 720, 430), fill="red")
    source = io.BytesIO()
    im.save(source, format="PNG")
    hashes = perceptual_hashes(source.getvalue())
    write_index(tmp_path, hashes, [0] * len(hashes), ["heldout-source"])
    smaller = io.BytesIO()
    im.resize((400, 300), Image.Resampling.LANCZOS).save(smaller, format="JPEG", quality=87)
    found = NearDuplicateIndex(tmp_path).lookup(perceptual_hashes(smaller.getvalue()))
    assert found["benchmark_path"] == "heldout-source"
    assert found["hamming_distance"] <= 3


def test_multi_index_finds_three_bit_changes_in_three_distinct_pieces(tmp_path):
    original = 0x123456789ABCDEF0
    write_index(tmp_path, [original], [0], ["benchmark"])
    index = NearDuplicateIndex(tmp_path)
    assert index.lookup([original ^ (1 << 4) ^ (1 << 20) ^ (1 << 37)])["hamming_distance"] == 3
    assert index.lookup([original ^ 0xFFFFFFFFFFFFFFFF]) is None
