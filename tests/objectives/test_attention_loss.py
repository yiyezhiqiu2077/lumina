from types import MethodType, SimpleNamespace

import torch
from torch import nn

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
    assert auxiliary.shape == (5,)
    assert auxiliary[4].item() == 1
    auxiliary[0].backward()
    assert q.grad is not None and q.grad.abs().sum() > 0
    assert k.grad is not None and k.grad.abs().sum() > 0


def test_attention_auxiliary_uses_pre_rope_qk(monkeypatch):
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

    def fake_auxiliary(q, k, *args):
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
    assert torch.equal(captured["q"], expected_q)
    assert torch.equal(captured["k"], expected_k)
    assert torch.equal(captured["attention_q"], expected_q + 100.0)
    assert torch.equal(captured["attention_k"], expected_k + 200.0)
