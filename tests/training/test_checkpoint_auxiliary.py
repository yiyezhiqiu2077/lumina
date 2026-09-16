import torch

from models.lumina.configuration_llada import LLaDAConfig
from models.lumina.modeling_xllmx_dimoo import LLaDAForMultiModalGeneration


def test_whole_layer_checkpoint_returns_one_auxiliary_per_layer_with_gradients():
    config = LLaDAConfig(
        d_model=64,
        n_heads=4,
        n_kv_heads=4,
        n_layers=2,
        mlp_hidden_size=128,
        vocab_size=128,
        embedding_size=128,
        max_sequence_length=32,
        block_type="llama",
        activation_type="silu",
        weight_tying=False,
        rope=True,
        attention_dropout=0.0,
        residual_dropout=0.0,
        embedding_dropout=0.0,
        init_device="cpu",
    )
    model = LLaDAForMultiModalGeneration(config)
    for parameter in model.parameters():
        if parameter.is_floating_point():
            torch.nn.init.normal_(parameter, 0, 0.02)
    model.model.set_activation_checkpointing("whole_layer")
    tokens = torch.randint(0, 128, (1, 12))
    instruction = torch.zeros(1, 12, dtype=torch.bool)
    instruction[:, 1:4] = True
    source = torch.zeros(1, 12, dtype=torch.bool)
    source[:, 5:11] = True
    edit = torch.zeros(1, 12, dtype=torch.bool)
    edit[:, 7:9] = True
    output = model.model(
        input_ids=tokens,
        attention_supervision_layers=[0, 1],
        instruction_token_mask=instruction,
        source_spatial_mask=source,
        source_edit_mask=edit,
        attention_active=torch.tensor([True]),
    )
    assert output.attention_auxiliary.shape == (2, 5)
    assert output.attention_auxiliary[:, 4].tolist() == [1.0, 1.0]
    output.attention_auxiliary[:, 0].mean().backward()
    for block in model.model.transformer.blocks:
        assert block.q_proj.weight.grad is not None
        assert block.k_proj.weight.grad is not None
