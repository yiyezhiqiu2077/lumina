from types import MethodType, SimpleNamespace

import torch
from torch import nn
import pytest

from models.lumina import modeling_llada
from models.lumina.modeling_llada import LLaDABlock
from models.objectives.attention import layer_attention_auxiliary, spatial_cross_entropy


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
    assert auxiliary.shape == (8,)
    assert auxiliary[4].item() == 1
    auxiliary[0].backward()
    assert q.grad is not None and q.grad.abs().sum() > 0
    assert k.grad is not None and k.grad.abs().sum() > 0


def test_attention_auxiliary_uses_post_rope_qk_by_default(monkeypatch):
    block = object.__new__(LLaDABlock)
    nn.Module.__init__(block)
    block.config = SimpleNamespace(
        n_heads=2,
        effective_n_kv_heads=2,
        rope=True,
        attention_dropout=0.0,
    )
    block.use_cache = False
    block.q_norm = None
    block.k_norm = None
    block.attn_out = nn.Identity()

    class FakeRoPE(nn.Module):
        def forward(self, q, k, q_mask=None):
            return q + 100.0, k + 200.0

    block.rotary_emb = FakeRoPE()
    captured = {}

    def fake_auxiliary(q, k, *args, **kwargs):
        captured["q"] = q.detach().clone()
        captured["k"] = k.detach().clone()
        return q.sum() * 0 + torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0])

    def fake_attention(self, q, k, v, **kwargs):
        captured["attention_q"] = q.detach().clone()
        captured["attention_k"] = k.detach().clone()
        return torch.zeros_like(q)

    monkeypatch.setattr(modeling_llada, "layer_attention_auxiliary", fake_auxiliary)
    block._scaled_dot_product_attention = MethodType(fake_attention, block)
    q = torch.arange(24, dtype=torch.float32).reshape(1, 3, 8)
    k = q + 1
    v = q + 2
    mask = torch.ones(1, 3, dtype=torch.bool)
    block.attention(
        q,
        k,
        v,
        instruction_token_mask=mask,
        source_spatial_mask=mask,
        source_edit_mask=mask,
        attention_active=torch.tensor([True]),
    )
    expected_q = q.view(1, 3, 2, 4).transpose(1, 2)
    expected_k = k.view(1, 3, 2, 4).transpose(1, 2)
    assert torch.equal(captured["q"], expected_q + 100.0)
    assert torch.equal(captured["k"], expected_k + 200.0)
    assert torch.equal(captured["attention_q"], expected_q + 100.0)
    assert torch.equal(captured["attention_k"], expected_k + 200.0)


def test_attention_auxiliary_can_use_pre_rope_qk(monkeypatch):
    block = object.__new__(LLaDABlock)
    nn.Module.__init__(block)
    block.config = SimpleNamespace(n_heads=2, effective_n_kv_heads=2, rope=True, attention_dropout=0.0)
    block.use_cache = False
    block.q_norm = block.k_norm = None
    block.attn_out = nn.Identity()

    class FakeRoPE(nn.Module):
        def forward(self, q, k, q_mask=None):
            return q + 100.0, k + 200.0

    block.rotary_emb = FakeRoPE()
    captured = {}
    monkeypatch.setattr(modeling_llada, "layer_attention_auxiliary", lambda q, k, *args, **kwargs: captured.update(q=q, k=k) or q.sum() * 0 + torch.tensor([1., 0., 0., 0., 1., 0., 0., 0.]))
    block._scaled_dot_product_attention = MethodType(lambda self, q, k, v, **kwargs: torch.zeros_like(q), block)
    q = torch.arange(24, dtype=torch.float32).reshape(1, 3, 8)
    mask = torch.ones(1, 3, dtype=torch.bool)
    block.attention(q, q + 1, q + 2, instruction_token_mask=mask, source_spatial_mask=mask, source_edit_mask=mask, attention_active=torch.tensor([True]), attention_qk_stage="pre_rope")
    assert torch.equal(captured["q"], q.view(1, 3, 2, 4).transpose(1, 2))


def test_region_mass_and_normalized_mask_ce_statistics():
    q = torch.tensor([[[[2.0], [0.0], [0.0], [0.0]]]], requires_grad=True)
    k = torch.ones_like(q, requires_grad=True)
    instruction = torch.tensor([[True, False, False, False]])
    source = torch.tensor([[True, True, True, True]])
    edit = torch.tensor([[True, True, False, False]])
    normalized = layer_attention_auxiliary(q, k, instruction, source, edit, torch.tensor([True]))
    region = layer_attention_auxiliary(q, k, instruction, source, edit, torch.tensor([True]), mode="region_mass")
    assert torch.allclose(normalized[6], torch.log(torch.tensor(2.0)))
    assert torch.allclose(normalized[7], normalized[0] - normalized[6])
    assert torch.allclose(region[0], -(region[1] + 1e-8).log())


def test_region_mass_uses_exact_raw_probability_mass():
    # q @ k gives equal source scores, so selecting one of two source tokens
    # has P_G=0.5 exactly.  Selecting both has P_G=1 exactly.
    q = torch.ones(1, 1, 2, 1, requires_grad=True)
    k = torch.zeros(1, 1, 2, 1, requires_grad=True)
    instruction = torch.tensor([[True, False]])
    source = torch.tensor([[True, True]])
    active = torch.tensor([True])
    half = layer_attention_auxiliary(
        q, k, instruction, source, torch.tensor([[True, False]]), active, mode="region_mass"
    )
    full = layer_attention_auxiliary(
        q, k, instruction, source, torch.tensor([[True, True]]), active, mode="region_mass"
    )
    assert half[1].item() == pytest.approx(0.5, abs=1e-7)
    assert half[0].item() == pytest.approx(-torch.log(torch.tensor(0.5)).item(), abs=1e-7)
    assert full[1].item() == pytest.approx(1.0, abs=1e-7)
    assert full[0].item() == pytest.approx(0.0, abs=1e-7)
