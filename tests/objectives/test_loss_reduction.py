import torch
import torch.nn.functional as F

from models.lumina.modeling_xllmx_dimoo import generation_ce_and_z_loss
from models.objectives.reduction import reduce_supervised_values


def test_sample_mean_and_token_mean_differ_with_unequal_counts():
    values = torch.tensor([[1.0, 0.0, 0.0], [3.0, 3.0, 3.0]])
    valid = torch.tensor([[True, False, False], [True, True, True]])
    assert reduce_supervised_values(values, valid, "sample_mean") == torch.tensor(2.0)
    assert reduce_supervised_values(values, valid, "token_mean") == torch.tensor(2.5)


def test_generation_ce_and_z_loss_match_explicit_per_sample_reduction():
    logits = torch.tensor(
        [
            [[2.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            [[0.0, 2.0], [0.0, 2.0], [0.0, 2.0]],
        ],
        requires_grad=True,
    )
    labels = torch.tensor([[0, -100, -100], [0, 0, 0]])
    ce, z_loss, count, max_abs_logit = generation_ce_and_z_loss(
        logits,
        labels,
        "sample_mean",
    )
    first = F.cross_entropy(logits[0, :1].float(), torch.tensor([0]))
    second = F.cross_entropy(logits[1].float(), torch.tensor([0, 0, 0]))
    expected_ce = (first + second) / 2
    valid_logits = torch.cat((logits[0, :1], logits[1])).float()
    token_z = torch.logsumexp(valid_logits, dim=-1).square()
    expected_z = (token_z[:1].mean() + token_z[1:].mean()) / 2
    assert torch.allclose(ce, expected_ce)
    assert torch.allclose(z_loss, expected_z)
    assert count.item() == 4
    assert max_abs_logit.item() == 2.0
    (ce + 1e-5 * z_loss).backward()
    assert torch.isfinite(logits.grad).all()


def test_token_mean_matches_flattened_cross_entropy_and_z_loss():
    logits = torch.tensor(
        [[[2.0, 0.0], [0.0, 0.0]], [[0.0, 2.0], [0.0, 2.0]]],
        requires_grad=True,
    )
    labels = torch.tensor([[0, -100], [0, 0]])
    ce, z_loss, count, _ = generation_ce_and_z_loss(logits, labels, "token_mean")
    valid_logits = logits[labels != -100].float()
    valid_labels = labels[labels != -100]
    assert torch.allclose(ce, F.cross_entropy(valid_logits, valid_labels))
    assert torch.allclose(z_loss, torch.logsumexp(valid_logits, dim=-1).square().mean())
    assert count.item() == 3


def test_empty_supervision_is_finite_and_differentiable():
    logits = torch.randn(2, 3, 5, requires_grad=True)
    labels = torch.full((2, 3), -100)
    ce, z_loss, count, max_abs_logit = generation_ce_and_z_loss(
        logits,
        labels,
        "sample_mean",
    )
    assert ce.item() == z_loss.item() == max_abs_logit.item() == 0.0
    assert count.item() == 0
    (ce + z_loss).backward()
    assert torch.equal(logits.grad, torch.zeros_like(logits))
