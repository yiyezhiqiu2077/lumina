from dataset.geometry import expected_token_grid, sample_shared_geometry


def test_shared_geometry_is_deterministic_square_and_has_a_32_grid():
    first = sample_shared_geometry((1536, 1024), 42, 512)
    assert first == sample_shared_geometry((1536, 1024), 42, 512)
    assert (first.crop_width, first.crop_height) == (512, 512)
    assert expected_token_grid(first) == (32, 32)
