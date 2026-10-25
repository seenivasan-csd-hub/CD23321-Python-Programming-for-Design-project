"""Source-linked extension guidance kept separate from image model output."""

import json
from pathlib import Path


GUIDANCE_PATH = Path(__file__).resolve().parents[1] / "data" / "disease_guidance.json"
_GUIDANCE_DATA = json.loads(GUIDANCE_PATH.read_text(encoding="utf-8"))
DISEASE_GUIDANCE = _GUIDANCE_DATA["guidance"]
PLANTDOC_ALIASES = {key.casefold(): value for key, value in _GUIDANCE_DATA["plantdoc_aliases"].items()}
GENERAL_GUIDANCE = {
    **DISEASE_GUIDANCE["uncertain"],
    "name": "No source-verified guidance for this model class",
    "crop": "Crop and class need confirmation",
    "symptoms": "FarmIQ has no source-verified description for this exact class. A model label is not a diagnosis.",
    "next_steps": "Retake a clear photo and ask a local TNAU or KVK extension professional to inspect the plant before choosing any treatment.",
    "source_title": "Tamil Nadu Agricultural University Agritech Portal",
    "source_url": "https://agritech.tnau.ac.in/",
}


def guidance_for_prediction(prediction):
    if not prediction:
        return {}
    label = str(prediction.get("raw_label", ""))
    key = PLANTDOC_ALIASES.get(label.casefold(), label)
    if prediction.get("uncertain") or prediction.get("crop_mismatch"):
        return DISEASE_GUIDANCE["uncertain"]
    return DISEASE_GUIDANCE.get(key, GENERAL_GUIDANCE)
