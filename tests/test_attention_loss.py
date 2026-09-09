import torch

from attention_supervision.attention_loss import layer_attention_auxiliary, spatial_cross_entropy


def test_inside_attention_has_lower_loss_than_outside_attention():
    mask = torch.tensor([1.0, 1.0, 0.0, 0.0])
    inside = torch.tensor([0.49, 0.49, 0.01, 0.01], requires_grad=True)
    outside = torch.tensor([0.01, 0.01, 0.49, 0.49], requires_grad=True)
    active = torch.tensor([True])
    inside_loss, _ = spatial_cross_entropy([inside], [mask], active)
    outside_loss, _ = spatial_cross_entropy([outside], [mask], active)
    assert inside_loss < outside_loss
    inside_loss.backward()
    assert inside.grad is not None


def test_empty_or_unconditional_mask_has_zero_loss():
    attention = torch.full((4,), 0.25, requires_grad=True)
    empty_loss, stats = spatial_cross_entropy([attention], [torch.zeros(4)], torch.tensor([True]))
    unconditional_loss, _ = spatial_cross_entropy([attention], [torch.ones(4)], torch.tensor([False]))
    assert empty_loss.item() == 0.0
    assert unconditional_loss.item() == 0.0
    assert stats["active_samples"].item() == 0


def test_layer_auxiliary_has_expected_metrics_and_gradients():
    torch.manual_seed(17)
    q = torch.randn(2, 4, 12, 8, requires_grad=True)
    k = torch.randn(2, 4, 12, 8, requires_grad=True)
    instruction = torch.zeros(2, 12, dtype=torch.bool)
    instruction[:, 1:4] = True
    source = torch.zeros(2, 12, dtype=torch.bool)
    source[:, 5:11] = True
    edit = torch.zeros(2, 12, dtype=torch.bool)
    edit[:, 7:9] = True
    auxiliary = layer_attention_auxiliary(q, k, instruction, source, edit, torch.tensor([True, False]))
    assert auxiliary.shape == (5,)
    assert auxiliary[4].item() == 1
    auxiliary[0].backward()
    assert q.grad is not None and q.grad.abs().sum() > 0
    assert k.grad is not None and k.grad.abs().sum() > 0
