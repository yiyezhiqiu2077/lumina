import pytest
import torch

from lumina_dimoo.models import MagicBrushModelOutput
from lumina_dimoo.training.objective import compose_total_loss, run_model_for_objective, validate_objective


class FakeModel:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        auxiliary = torch.tensor([[2.0, 0.2, 0.3, 0.4, 1.0]], requires_grad=True) if "attention_supervision_layers" in kwargs else None
        return MagicBrushModelOutput(
            generation_loss=torch.tensor(3.0, requires_grad=True),
            logits=torch.zeros(1, 1, 4, requires_grad=True),
            labels=torch.tensor([[0]]),
            attention_auxiliary=auxiliary,
        )


def test_ce_is_exactly_generation_loss_and_requests_no_auxiliary():
    model = FakeModel()
    result = run_model_for_objective(model, objective="ce", input_ids=[[1]], labels=[[1]])
    assert compose_total_loss("ce", result.output.generation_loss) is result.output.generation_loss
    assert "attention_supervision_layers" not in model.calls[0]


def test_attention_does_not_call_gce_and_uses_expected_sum(monkeypatch):
    from lumina_dimoo.objectives.gce import GCEObjective

    monkeypatch.setattr(GCEObjective, "forward", lambda *_: (_ for _ in ()).throw(AssertionError("GCE called")))
    model = FakeModel()
    result = run_model_for_objective(
        model,
        objective="attention",
        input_ids=[[1]], labels=[[1]],
        attention_layers=[24], attention_masks={"instruction_token_mask": [[True]], "source_spatial_mask": [[True]], "source_edit_mask": [[True]], "attention_active": [True]},
        gce_objective=GCEObjective.__new__(GCEObjective),
    )
    raw = result.attention_auxiliary.mean(0)[0]
    total = compose_total_loss("attention", result.output.generation_loss, attention_loss=raw, attention_weight=0.1)
    assert torch.allclose(total, torch.tensor(3.2))
    assert "attention_supervision_layers" in model.calls[0]


def test_gce_does_not_request_attention_and_uses_expected_sum():
    class FakeGCE:
        def __call__(self, logits, labels):
            assert logits.shape[-1] == 4 and labels.shape == (1, 1)
            return torch.tensor(5.0, requires_grad=True), {"gce_loss": torch.tensor(5.0)}

    model = FakeModel()
    result = run_model_for_objective(model, objective="gce", input_ids=[[1]], labels=[[1]], gce_objective=FakeGCE())
    total = compose_total_loss("gce", result.output.generation_loss, gce_loss=result.gce_loss, gce_weight=1.0)
    assert torch.allclose(total, torch.tensor(8.0))
    assert "attention_supervision_layers" not in model.calls[0]


def test_invalid_or_mixed_objective_is_rejected():
    with pytest.raises(ValueError):
        validate_objective("attention+gce")
    with pytest.raises(ValueError):
        compose_total_loss("ce+gce", torch.tensor(1.0))
