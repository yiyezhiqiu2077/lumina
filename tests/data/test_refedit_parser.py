from io import BytesIO

from PIL import Image

from dataset.refedit import audit_row_reason, decode_image, decode_mask


def _png(mode: str) -> bytes:
    image = Image.new(mode, (8, 6), 255)
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def test_refedit_qc_gate_and_embedded_image_decoding():
    accepted = {
        "prefilter_verdict": "PASS",
        "grounding_status": "OK",
        "mask_qc_flag": "OK",
        "instruction": "make it blue",
        "area_frac": 0.1,
    }
    assert audit_row_reason(accepted) is None
    assert audit_row_reason({**accepted, "grounding_status": "FAILED"}) == "grounding_status='FAILED'"
    assert decode_image({"bytes": _png("RGB"), "path": None}, mode="RGB").size == (8, 6)
    assert decode_mask(_png("L")).size == (8, 6)
