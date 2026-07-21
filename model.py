"""
model.py — Custom fine-tuned mBERT sentiment classification
and single-aspect extraction for SentiMalay.
"""

import os
import re
from functools import lru_cache
from typing import Any

import torch
from huggingface_hub import hf_hub_download
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

# Hugging Face repository containing the trained checkpoint.
HF_REPO_ID = "nvjlaa/mBERT"
HF_MODEL_FILENAME = "mbert_sentiment_model.pkl"

# Exactly one of these aspects is assigned to each comment.
ASPECTS = {
    "Taste / Rasa": [
        "sedap", "lazat", "lezat", "rasa", "masin", "manis", "masam",
        "pahit", "pedas", "tawar", "enak", "nyaman", "lemak", "taste",
        "delicious", "flavour", "flavor", "yummy", "tasteless", "bland",
        "sweet", "salty", "sour", "spicy", "bitter", "kurang sedap",
        "tak sedap", "sangat sedap", "memang sedap", "sedap sangat",
    ],
    "Ingredients / Bahan": [
        "bahan", "sukatan", "resepi", "resipi", "ramuan", "ganti",
        "ingredient", "recipe", "substitute", "measurement", "quantity",
        "amount", "portion", "spice", "rempah", "santan", "minyak",
        "garam", "gula", "tepung", "sayur", "daging", "ayam", "ikan",
        "udang", "telur",
    ],
    "Cooking Steps / Langkah": [
        "cara", "langkah", "kaedah", "teknik", "proses", "mudah", "susah",
        "sukar", "method", "step", "process", "easy", "hard", "difficult",
        "simple", "follow", "instructions", "tutorial", "guide", "demo",
        "ikut", "faham", "jelas", "peringkat", "prosedur", "arahan",
    ],
    "Time / Masa": [
        "lama", "cepat", "lambat", "minit", "jam", "masa", "duration",
        "quick", "slow", "fast", "long", "short", "minute", "hour",
        "time", "tempoh", "sekejap", "sebentar", "berapa lama",
    ],
    "Presentation / Persembahan": [
        "cantik", "comel", "menarik", "kemas", "presentation", "plating",
        "look", "beautiful", "nice", "neat", "video", "quality", "visual",
        "gambar", "foto", "warna", "colour", "color", "hiasan", "garnish",
    ],
    "Texture / Tekstur": [
        "lembut", "keras", "rangup", "gebu", "moist", "crispy", "crunchy",
        "soft", "fluffy", "dry", "wet", "texture", "tekstur", "kenyal",
        "garing", "berderai", "halus", "kasar", "licin",
    ],
}

# Used when two aspects obtain the same score.
ASPECT_PRIORITY = [
    "Taste / Rasa",
    "Ingredients / Bahan",
    "Cooking Steps / Langkah",
    "Time / Masa",
    "Presentation / Persembahan",
    "Texture / Tekstur",
]


def _normalise_id2label(raw_mapping: Any) -> dict[int, str]:
    """Convert checkpoint label keys to integer class IDs."""
    if not isinstance(raw_mapping, dict):
        return {}

    mapping: dict[int, str] = {}
    for key, value in raw_mapping.items():
        try:
            mapping[int(key)] = str(value)
        except (TypeError, ValueError):
            continue
    return mapping


def _normalise_label2id(raw_mapping: Any) -> dict[str, int]:
    """Convert checkpoint label values to integer class IDs."""
    if not isinstance(raw_mapping, dict):
        return {}

    mapping: dict[str, int] = {}
    for key, value in raw_mapping.items():
        try:
            mapping[str(key)] = int(value)
        except (TypeError, ValueError):
            continue
    return mapping


def _load_torch_checkpoint(checkpoint_path: str) -> dict:
    """
    Load the trusted checkpoint created during training.

    The fallback keeps compatibility with older PyTorch versions that do not
    accept the weights_only argument.
    """
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "The downloaded .pkl file is not the expected checkpoint dictionary."
        )

    required = {"model_state_dict", "model_name"}
    missing = required.difference(checkpoint.keys())
    if missing:
        raise KeyError(
            "Checkpoint is missing required field(s): "
            + ", ".join(sorted(missing))
        )

    return checkpoint


@lru_cache(maxsize=1)
def load_model() -> dict:
    """
    Download the trained checkpoint from Hugging Face, reconstruct the mBERT
    architecture, load the fine-tuned weights, and cache the result.
    """
    checkpoint_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_MODEL_FILENAME,
        repo_type="model",
        token=os.getenv("HF_TOKEN") or None,
    )

    checkpoint = _load_torch_checkpoint(checkpoint_path)

    model_name = str(checkpoint["model_name"])
    id2label = _normalise_id2label(checkpoint.get("id2label"))
    label2id = _normalise_label2id(checkpoint.get("label2id"))

    if not id2label and label2id:
        id2label = {class_id: label for label, class_id in label2id.items()}

    if not label2id and id2label:
        label2id = {label: class_id for class_id, label in id2label.items()}

    if not id2label:
        # Fallback only. The checkpoint's own mapping is preferred.
        id2label = {0: "Negative", 1: "Neutral", 2: "Positive"}
        label2id = {label: class_id for class_id, label in id2label.items()}

    num_labels = len(id2label)

    # Build the model architecture from its configuration only.
    # The trained weights are then loaded from model_state_dict.
    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
    model = AutoModelForSequenceClassification.from_config(config)

    state_dict = checkpoint["model_state_dict"]

    # Support checkpoints created with torch.nn.DataParallel.
    if state_dict and all(str(key).startswith("module.") for key in state_dict):
        state_dict = {
            str(key).removeprefix("module."): value
            for key, value in state_dict.items()
        }

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return {
        "model": model,
        "tokenizer": tokenizer,
        "id2label": id2label,
        "max_len": int(checkpoint.get("max_len", 512)),
    }


def preprocess(text: str) -> str:
    """Clean a raw YouTube comment using the same style as the application."""
    text = re.sub(r"http\S+|www\S+", "", str(text))
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^\w\s',.!?]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def _canonical_sentiment(raw_label: str, predicted_id: int) -> str:
    """Return one of Positive, Neutral, or Negative."""
    label = str(raw_label).strip().lower()

    if "positive" in label:
        return "Positive"
    if "neutral" in label:
        return "Neutral"
    if "negative" in label:
        return "Negative"

    # Fallback for LABEL_0/LABEL_1/LABEL_2-style labels.
    fallback = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return fallback.get(predicted_id, str(raw_label))


def predict_sentiment(clean_text: str, classifier: dict) -> tuple[str, float]:
    """Run inference using the fine-tuned model loaded from the checkpoint."""
    model = classifier["model"]
    tokenizer = classifier["tokenizer"]
    id2label = classifier["id2label"]
    max_len = classifier["max_len"]

    encoded = tokenizer(
        clean_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_len,
    )

    with torch.inference_mode():
        logits = model(**encoded).logits
        probabilities = torch.softmax(logits, dim=-1)[0]

    predicted_id = int(torch.argmax(probabilities).item())
    confidence = float(probabilities[predicted_id].item())
    raw_label = id2label.get(predicted_id, f"LABEL_{predicted_id}")

    return _canonical_sentiment(raw_label, predicted_id), round(confidence, 4)


def _keyword_matches(text: str, keyword: str) -> bool:
    """Match a complete word or phrase instead of a substring."""
    pattern = rf"(?<!\w){re.escape(keyword.lower())}(?!\w)"
    return re.search(pattern, text.lower()) is not None


def extract_aspect(text: str) -> str:
    """
    Assign exactly one aspect to one comment.

    The aspect with the highest keyword score is selected. Multi-word phrases
    receive two points and single words receive one point. Priority order is
    used only when scores are tied.
    """
    scores: dict[str, int] = {}

    for aspect in ASPECT_PRIORITY:
        matched = {
            keyword.lower()
            for keyword in ASPECTS[aspect]
            if _keyword_matches(text, keyword)
        }
        scores[aspect] = sum(2 if " " in keyword else 1 for keyword in matched)

    best_aspect = max(
        ASPECT_PRIORITY,
        key=lambda aspect: (
            scores[aspect],
            -ASPECT_PRIORITY.index(aspect),
        ),
    )

    return best_aspect if scores[best_aspect] > 0 else "General"


def analyse_comment(text: str, classifier: dict) -> dict:
    """
    Full processing flow:
    preprocess -> trained mBERT prediction -> exactly one aspect.
    """
    clean = preprocess(text)

    if not clean:
        raise ValueError("The comment is empty after preprocessing.")

    sentiment, confidence = predict_sentiment(clean, classifier)
    selected_aspect = extract_aspect(clean)

    # Keep 'aspects' as a one-item list so the existing app.py dashboard,
    # history, table, and CSV code continue to work without major changes.
    return {
        "original": text,
        "clean": clean,
        "sentiment": sentiment,
        "confidence": confidence,
        "aspects": [selected_aspect],
    }
