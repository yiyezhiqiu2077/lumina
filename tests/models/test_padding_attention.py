import torch

from model.modeling_llada import LLaDABlock


def _block_without_flash():
    block = object.__new__(LLaDABlock)
    object.__setattr__(block, "flash_attn_func", None)
    return block


def test_padding_key_value_cannot_affect_real_query():
    block = _block_without_flash()
    q = torch.tensor([[[[1.0, 0.0]]]])
    k = torch.tensor([[[[1.0, 0.0], [0.0, 1.0], [20.0, 0.0]]]])
    v = torch.tensor([[[[2.0, 0.0], [0.0, 4.0], [1000.0, 1000.0]]]])
    mask = torch.tensor([[[[True, True, False]]]])
    padded = block._scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False)
    reference = block._scaled_dot_product_attention(q, k[:, :, :2], v[:, :, :2], attn_mask=None, is_causal=False)
    assert torch.allclose(padded, reference, atol=1e-6, rtol=1e-6)


def test_all_valid_mask_matches_unmasked_noncausal_attention():
    torch.manual_seed(4)
    block = _block_without_flash()
    q = torch.randn(2, 4, 3, 8)
    k = torch.randn(2, 4, 3, 8)
    v = torch.randn(2, 4, 3, 8)
    masked = block._scaled_dot_product_attention(q, k, v, attn_mask=torch.ones(2, 1, 3, 3, dtype=torch.bool), is_causal=False)
    plain = block._scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=False)
    assert torch.allclose(masked, plain, atol=1e-6, rtol=1e-6)
