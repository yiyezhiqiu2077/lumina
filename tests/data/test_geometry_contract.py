from PIL import Image

from dataset.geometry import (
    apply_shared_geometry, expected_token_grid, mask_retention, sample_shared_geometry, valid_crop_sizes,
)


def test_lumina_buckets_choose_square_landscape_and_portrait_deterministically():
    square = sample_shared_geometry((1024, 1024), 42)
    landscape = sample_shared_geometry((1536, 1024), 42)
    portrait = sample_shared_geometry((1024, 1536), 42)
    assert (square.crop_width, square.crop_height) == (512, 512)
    assert (landscape.crop_width, landscape.crop_height) == (608, 416)
    assert (portrait.crop_width, portrait.crop_height) == (416, 608)
    assert sample_shared_geometry((1536, 1024), 42) == landscape
    assert sample_shared_geometry((1536, 1024), 0).crop_left != sample_shared_geometry((1536, 1024), 1).crop_left
    assert (landscape.crop_width, landscape.crop_height) in valid_crop_sizes()
    assert expected_token_grid(landscape) == (26, 38)
    assert expected_token_grid(portrait) == (38, 26)


def test_shared_transform_preserves_alignment_and_mask_retention_reports_crop_loss():
    geometry = sample_shared_geometry((1600, 800), 3)
    source = Image.new("RGB", (1600, 800), "red")
    target = Image.new("RGB", (1600, 800), "blue")
    mask = Image.new("L", (1600, 800), 0)
    # A right-edge edit may fall outside the deterministic crop; it must be
    # recorded rather than moving the crop based on the mask.
    for x in range(1550, 1600):
        for y in range(100, 200):
            mask.putpixel((x, y), 255)
    transformed = [apply_shared_geometry(image, geometry, is_mask=index == 2) for index, image in enumerate((source, target, mask))]
    assert {image.size for image in transformed} == {(geometry.crop_width, geometry.crop_height)}
    retention = mask_retention(mask, geometry)
    assert retention["raw_mask_pixels"] > 0
    assert 0.0 <= retention["mask_retention_ratio"] <= 1.0


def test_edge_edit_that_falls_outside_crop_is_explicitly_empty_after_geometry():
    geometry = sample_shared_geometry((1536, 1024), 6)
    assert geometry.crop_left > 0
    mask = Image.new("L", (1536, 1024), 0)
    for x in range(4):
        for y in range(20, 40):
            mask.putpixel((x, y), 255)
    retention = mask_retention(mask, geometry)
    assert retention["raw_mask_pixels"] > 0
    assert retention["post_geometry_empty_mask"] is True
    assert retention["mask_retention_ratio"] == 0.0


def test_max_legal_bucket_fits_image_editing_sequence_with_remaining_text_allowance():
    max_bucket = max(valid_crop_sizes(), key=lambda size: size[0] * size[1])
    geometry = sample_shared_geometry(max_bucket, 0)
    height, width = expected_token_grid(geometry)
    # Each image is spatial tokens plus one newline per row. Source has BOI/EOI
    # and target has answer/BOI/EOI/end delimiters.
    image_editing_tokens = (height * (width + 1) + 2) + (height * (width + 1) + 4)
    text_allowance = 5120 - image_editing_tokens
    assert text_allowance >= 0
    assert image_editing_tokens + text_allowance == 5120
