import torch

from models.objectives.gce import GroupedCrossEntropyLoss


def make_loss():
    mapping = torch.tensor([[0, -1], [1, 2], [3, -1]])
    sizes = torch.tensor([1, 2, 1])
    token_to_cluster = torch.tensor([0, 1, 1, 2])
    return GroupedCrossEntropyLoss({4: {"token_to_cluster": token_to_cluster, "cluster_map": mapping, "cluster_sizes": sizes}})


def test_singleton_equals_visual_ce_and_padding_is_ignored():
    loss = make_loss()
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)
    value, _ = loss(logits, torch.tensor([0]))
    expected = torch.logsumexp(logits, -1).mean() - logits[:, 0].mean()
    assert torch.allclose(value, expected)
    value.backward()
    assert torch.isfinite(logits.grad).all()


def test_group_logits_have_expected_monotonicity_and_empty_is_finite():
    loss = make_loss()
    base = torch.zeros(1, 4)
    same = base.clone(); same[0, 2] = 4
    outside = base.clone(); outside[0, 3] = 4
    a, _ = loss(base, torch.tensor([1])); b, _ = loss(same, torch.tensor([1])); c, _ = loss(outside, torch.tensor([1]))
    assert b < a < c
    empty, _ = loss(torch.empty(0, 4, dtype=torch.bfloat16), torch.empty(0, dtype=torch.long))
    assert torch.isfinite(empty)
