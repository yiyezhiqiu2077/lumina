import random

from lumina_dimoo.data.magicbrush import stable_sample_seed


def test_sample_identity_seed_is_order_and_index_independent():
    first = stable_sample_seed(42, 3, "sample-abc")
    reordered = stable_sample_seed(42, 3, "sample-abc")
    assert first == reordered
    assert random.Random(first).sample(range(128), 8) == random.Random(reordered).sample(range(128), 8)


def test_epoch_and_global_seed_change_the_corruption_stream():
    base = stable_sample_seed(42, 3, "sample-abc")
    assert base != stable_sample_seed(42, 4, "sample-abc")
    assert base != stable_sample_seed(43, 3, "sample-abc")
    assert base != stable_sample_seed(42, 3, "sample-def")
