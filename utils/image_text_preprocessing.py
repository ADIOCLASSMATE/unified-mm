"""Freeze semantically safe 512px views before either text or VAE encoding."""

import hashlib
import io
import math

from PIL import Image, ImageCms, ImageOps


CRITICAL = {"counting", "ocr", "text", "relation", "spatial", "attribute_binding"}
VIEW_VERSION = "rgb512-safe-crop-v1"


def prepare_view(data: bytes, row: dict, min_short_side: int = 512, include_phashes: bool = False):
    with Image.open(io.BytesIO(data)) as source:
        if getattr(source, "n_frames", 1) != 1:
            raise ValueError("animated/multi-page source")
        if source.width * source.height > 80_000_000:
            raise ValueError("source exceeds 80 MP decode limit")
        orientation = source.getexif().get(274, 1)
        image = ImageOps.exif_transpose(source)
        icc = image.info.get("icc_profile")
        if icc:
            alpha = image.getchannel("A") if "A" in image.getbands() else None
            try:
                image = ImageCms.profileToProfile(
                    image, ImageCms.ImageCmsProfile(io.BytesIO(icc)),
                    ImageCms.createProfile("sRGB"), outputMode="RGB",
                )
            except (ImageCms.PyCMSError, OSError) as exc:
                raise ValueError("invalid or unsupported ICC profile") from exc
            if alpha is not None:
                image.putalpha(alpha)
        if "A" in image.getbands() or image.info.get("transparency") is not None:
            rgba = image.convert("RGBA")
            background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            image = Image.alpha_composite(background, rgba).convert("RGB")
        else:
            image = image.convert("RGB")
        width, height = image.size
        side = min(width, height)
        threshold = int(row.get("min_short_side", min_short_side))
        if threshold < 1 or side < threshold:
            raise ValueError(f"short side {side} < {threshold}")
        fit_pad = row.get("view_policy") == "fit_pad"
        if row.get("view_policy", "square_crop") not in {"square_crop", "fit_pad"}:
            raise ValueError("unknown view policy")
        if fit_pad and max(width, height) / side > 2.5:
            raise ValueError("full-frame padding would leave too little useful image area")
        boxes = row.get("required_boxes", [])
        normalized = row.get("required_boxes_normalized", [])
        if normalized:
            if boxes:
                raise ValueError("specify either pixel or normalized required boxes")
            boxes = []
            for box in normalized:
                if len(box) != 4 or not all(math.isfinite(float(v)) and 0 <= float(v) <= 1 for v in box):
                    raise ValueError("invalid normalized required box")
                boxes.append([float(v) * scale for v, scale in zip(box, (width, height, width, height))])
        if boxes and orientation != 1 and row.get("boxes_frame") != "exif":
            raise ValueError("boxes must be expressed in EXIF-normalized coordinates")
        if boxes:
            for box in boxes:
                if len(box) != 4 or not all(math.isfinite(float(v)) for v in box):
                    raise ValueError("invalid required box")
                x0, y0, x1, y1 = box
                if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
                    raise ValueError("required box lies outside the image")
            x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
            x1, y1 = max(b[2] for b in boxes), max(b[3] for b in boxes)
            low_x, high_x = max(0, math.ceil(x1 - side)), min(width - side, math.floor(x0))
            low_y, high_y = max(0, math.ceil(y1 - side)), min(height - side, math.floor(y0))
            if (low_x > high_x or low_y > high_y) and not fit_pad:
                raise ValueError("square crop cannot retain all required regions")
            left = min(high_x, max(low_x, (width - side) // 2))
            top = min(high_y, max(low_y, (height - side) // 2))
        else:
            critical = bool(CRITICAL.intersection(row.get("capabilities", [])))
            max_ratio = 1.15 if critical else 1.8
            if max(width, height) / side > max_ratio and not fit_pad:
                raise ValueError("wide view needs required_boxes or a separately reviewed crop")
            left, top = (width - side) // 2, (height - side) // 2
        crop = (left, top, left + side, top + side)
        hashes = []
        if include_phashes:
            from utils.image_near_duplicates import image_perceptual_hashes
            hashes = image_perceptual_hashes(image)
        content_box = None
        if fit_pad:
            crop = (0, 0, width, height)
            fitted = ImageOps.contain(image, (512, 512), Image.Resampling.LANCZOS)
            left, top = (512 - fitted.width) // 2, (512 - fitted.height) // 2
            image = Image.new("RGB", (512, 512), (127, 127, 127))
            image.paste(fitted, (left, top))
            content_box = [left, top, left + fitted.width, top + fitted.height]
        else:
            image = image.crop(crop).resize((512, 512), Image.Resampling.LANCZOS)
        output = io.BytesIO()
        text_image = bool({"ocr", "text"}.intersection(row.get("capabilities", [])))
        if text_image:
            image.save(output, format="PNG")
        else:
            image.save(output, format="JPEG", quality=95, subsampling=0)
        encoded = output.getvalue()
    metadata = {
        "view_version": "rgb512-full-frame-pad-v1" if fit_pad else VIEW_VERSION,
        "view_sha256": hashlib.sha256(encoded).hexdigest(),
        "source_sha256": hashlib.sha256(data).hexdigest(),
        "original_size": [width, height], "crop": list(crop),
        "upsampled": side < 512,
        "image_size": 512, "extension": "png" if text_image else "jpg",
    }
    if content_box is not None:
        metadata["content_box"] = content_box
        metadata["padding_rgb"] = [127, 127, 127]
    if include_phashes:
        from utils.image_near_duplicates import perceptual_hashes
        metadata["perceptual_hashes"] = sorted(set(hashes + perceptual_hashes(encoded)))
    return encoded, metadata
