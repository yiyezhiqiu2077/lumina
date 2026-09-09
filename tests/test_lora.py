import copy

import torch
from torch import nn

from attention_supervision.lora import LoRALinear, inject_lora


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(8, 8)
        self.k_proj = nn.Linear(8, 8)
        self.v_proj = nn.Linear(8, 8)
        self.attn_out = nn.Linear(8, 8)
        self.ff = nn.Linear(8, 8)


def test_zero_initialization_preserves_output_and_freezes_base():
    torch.manual_seed(7)
    model = TinyBlock()
    original = copy.deepcopy(model)
    report = inject_lora(model, rank=2, alpha=2, dropout=0)
    inputs = torch.randn(3, 8)
    assert torch.equal(model.q_proj(inputs), original.q_proj(inputs))
    assert len(report.matched_modules) == 4
    assert not model.ff.weight.requires_grad
    assert all("lora_" in name for name in report.trainable_names)


def test_q_and_k_lora_receive_gradients():
    torch.manual_seed(11)
    q = LoRALinear(nn.Linear(8, 8), rank=2, alpha=2, dropout=0)
    k = LoRALinear(nn.Linear(8, 8), rank=2, alpha=2, dropout=0)
    inputs = torch.randn(2, 5, 8)
    loss = (q(inputs) @ k(inputs).transpose(-1, -2)).softmax(-1)[..., 0].mean()
    loss.backward()
    assert q.lora_b.weight.grad is not None
    assert k.lora_b.weight.grad is not None
    assert torch.isfinite(q.lora_b.weight.grad).all()
    assert torch.isfinite(k.lora_b.weight.grad).all()
