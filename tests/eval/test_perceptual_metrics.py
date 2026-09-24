from __future__ import annotations

import numpy as np
from PIL import Image
import pytest
import torch

from evaluation.perceptual_metrics import (
    cosine_similarity,
    image_similarity_metrics,
    lpips_scores_from_spatial_map,
    mask_bbox,
    padded_mask_bbox,
    paired_roi_crops,
    spatial_lpips_masked_mean,
)


def test_mask_bbox_and_padding_clamp_are_correct():
    mask = np.zeros((10, 12), dtype=bool)
    mask[1:4, 2:6] = True
    assert mask_bbox(mask) == (2, 1, 6, 4)
    assert padded_mask_bbox(mask, padding_ratio=0.10) == (1, 0, 7, 5)

    edge = np.zeros((4, 4), dtype=bool)
    edge[:2, :2] = True
    assert padded_mask_bbox(edge, padding_ratio=0.50) == (0, 0, 3, 3)


def test_prediction_and_target_share_exact_same_roi_crop():
    mask = np.zeros((6, 8), dtype=bool)
    mask[2:4, 3:6] = True
    prediction = Image.fromarray(np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3))
    target = Image.fromarray(np.full((6, 8, 3), 127, dtype=np.uint8))
    prediction_roi, target_roi, box = paired_roi_crops(prediction, target, mask, padding_ratio=0.10)
    assert box == (2, 1, 7, 5)
    assert prediction_roi.size == target_roi.size == (5, 4)
    assert np.array_equal(np.asarray(prediction_roi), np.asarray(prediction)[1:5, 2:7])


def test_cosine_and_spatial_lpips_roi_mean_are_correct():
    assert cosine_similarity(torch.tensor([[2.0, 0.0]]), torch.tensor([[9.0, 0.0]])) == pytest.approx(1.0)
    spatial = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    mask = np.array([[False, True], [True, False]])
    assert spatial_lpips_masked_mean(spatial, mask) == pytest.approx(2.5)
    assert lpips_scores_from_spatial_map(spatial, mask) == {"full_lpips": pytest.approx(2.5), "roi_lpips": pytest.approx(2.5)}


def test_full_lpips_and_roi_lpips_use_distinct_reductions():
    spatial = torch.tensor([[[[1.0, 5.0], [9.0, 13.0]]]])
    mask = np.array([[True, False], [False, False]])
    assert lpips_scores_from_spatial_map(spatial, mask) == {"full_lpips": pytest.approx(7.0), "roi_lpips": pytest.approx(1.0)}


def test_empty_masks_fail_clearly():
    empty = np.zeros((3, 3), dtype=bool)
    with pytest.raises(ValueError, match="non-empty GT edit mask"):
        mask_bbox(empty)
    with pytest.raises(ValueError, match="non-empty GT edit mask"):
        spatial_lpips_masked_mean(torch.ones(1, 1, 2, 2), empty)


class _FakeSimilarity:
    def __init__(self) -> None:
        self.sizes: list[tuple[tuple[int, int], tuple[int, int]]] = []

    def similarity(self, prediction: Image.Image, target: Image.Image) -> float:
        self.sizes.append((prediction.size, target.size))
        return 1.0 if prediction.size == target.size else 0.0


def test_full_and_roi_similarity_use_same_crops():
    image = Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8))
    mask = np.zeros((10, 10), dtype=bool)
    mask[4:6, 4:6] = True
    metric = _FakeSimilarity()
    values = image_similarity_metrics(metric, image, image, mask, metric_name="dino_i", roi_padding_ratio=0.10)
    assert values == {"full_dino_i": 1.0, "roi_dino_i": 1.0}
    assert metric.sizes == [((10, 10), (10, 10)), ((4, 4), (4, 4))]
