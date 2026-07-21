"""
model.py — Load the trained joint mBERT aspect-sentiment model
from Hugging Face and analyse one comment at a time.
"""

import os
import pickle
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

# ============================================================
# HUGGING FACE MODEL LOCATION
# ============================================================
HF_REPO_ID = "nvjlaa/mBERT"
HF_MODEL_FILENAME = "mbert_aspect_sentiment_model.pkl"

# These are used only when the trained output label contains sentiment
# but does not contain an aspect name.
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
        "ingredient", "ingredients", "recipe", "substitute", "measurement",
        "quantity", "amount", "portion", "spice", "rempah", "santan",
        "minyak", "garam", "gula", "tepung", "sayur", "daging", "ayam",
        "ikan", "udang", "telur",
    ],
    "Cooking Steps / Langkah": [
        "cara", "langkah", "kaedah", "teknik", "proses", "mudah", "susah",
        "sukar", "method", "step", "steps", "process", "easy", "hard",
        "difficult", "simple", "follow", "instructions", "tutorial", "guide",
        "demo", "ikut", "faham", "jelas", "peringkat", "prosedur", "arahan",
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

ASPECT_PRIORITY = [
    "Taste / Rasa",
    "Ingredients / Bahan",
    "Cooking Steps / Langkah",
    "Time / Masa",
    "Presentation / Persembahan",
    "Texture / Tekstur",
]

ASPECT_ALIASES = {
    "taste": "Taste / Rasa",
    "rasa": "Taste / Rasa",
    "ingredient": "Ingredients / Bahan",
    "ingredients": "Ingredients / Bahan",
    "bahan": "Ingredients / Bahan",
    "cooking step": "Cooking Steps / Langkah",
    "cooking steps": "Cooking Steps / Langkah",
    "step": "Cooking Steps / Langkah",
    "steps": "Cooking Steps / Langkah",
    "langkah": "Cooking Steps / Langkah",
    "time": "Time / Masa",
    "masa": "Time / Masa",
    "presentation": "Presentation / Persembahan",
    "persembahan": "Presentation / Persembahan",
    "texture": "Texture / Tekstur",
    "tekstur": "Texture / Tekstur",
    "general": "General",
}


# ============================================================
# CHECKPOINT LOADING
# ============================================================
def _normalise_id2label(raw_mapping: Any) -> dict[int, str]:
    """Convert label mapping keys to integers."""
    if not isinstance(raw_mapping, dict):
        return {}

    output: dict[int, str] = {}

    for key, value in raw_mapping.items():
        try:
            output[int(key)] = str(value)
        except (TypeError, ValueError):
            continue

    return output


def _normalise_label2id(raw_mapping: Any) -> dict[str, int]:
    """Convert label mapping values to integers."""
    if not isinstance(raw_mapping, dict):
        return {}

    output: dict[str, int] = {}

    for key, value in raw_mapping.items():
        try:
            output[str(key)] = int(value)
        except (TypeError, ValueError):
            continue

    return output


def _mapping_from_classes(classes: Any) -> dict[int, str]:
    """Create id2label from a list, NumPy array, or label encoder."""
    if classes is None:
        return {}

    if hasattr(classes, "classes_"):
        classes = classes.classes_

    try:
        values = list(classes)
    except TypeError:
        return {}

    return {index: str(value) for index, value in enumerate(values)}


def _load_artifact(file_path: str) -> Any:
    """
    Load a trusted model artifact.

    It first tries torch.load(), then pickle.load(). This supports checkpoints
    saved using either torch.save(...) or pickle.dump(...).
    """
    with open(file_path, "rb") as file:
        first_bytes = file.read(200)

    if first_bytes.startswith(b"version https://git-lfs.github.com/spec/v1"):
        raise RuntimeError(
            "Only a Git LFS pointer was downloaded instead of the real model. "
            "Add `hf-xet` to requirements.txt and reboot the Streamlit app."
        )

    torch_error = None

    try:
        try:
            return torch.load(
                file_path,
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            return torch.load(file_path, map_location="cpu")
    except Exception as error:
        torch_error = error

    try:
        with open(file_path, "rb") as file:
            return pickle.load(file)
    except Exception as pickle_error:
        raise RuntimeError(
            "The Hugging Face model file could not be loaded. "
            f"torch.load error: {torch_error}. "
            f"pickle.load error: {pickle_error}."
        ) from pickle_error


def _find_value(mapping: dict, possible_keys: list[str], default=None):
    """Return the first existing value from several possible checkpoint keys."""
    for key in possible_keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def _build_runtime(artifact: Any) -> dict:
    """
    Convert the downloaded artifact into a runtime containing:
    model, tokenizer, id2label, and max_len.
    """

    # --------------------------------------------------------
    # Case 1: checkpoint dictionary
    # --------------------------------------------------------
    if isinstance(artifact, dict):
        # The file may already contain complete model and tokenizer objects.
        complete_model = _find_value(
            artifact,
            ["model", "trained_model", "classifier_model"],
        )
        complete_tokenizer = _find_value(
            artifact,
            ["tokenizer", "trained_tokenizer"],
        )

        if isinstance(complete_model, torch.nn.Module):
            model = complete_model
            model.eval()

            model_name = _find_value(
                artifact,
                [
                    "model_name",
                    "base_model_name",
                    "pretrained_model_name",
                    "tokenizer_name",
                ],
                getattr(getattr(model, "config", None), "_name_or_path", None),
            )

            tokenizer = complete_tokenizer
            if tokenizer is None:
                if not model_name:
                    raise KeyError(
                        "The checkpoint contains a model but no tokenizer or "
                        "base model name."
                    )
                tokenizer = AutoTokenizer.from_pretrained(str(model_name))

            id2label = _normalise_id2label(
                _find_value(artifact, ["id2label", "id_to_label"], {})
            )

            if not id2label:
                id2label = _normalise_id2label(
                    getattr(getattr(model, "config", None), "id2label", {})
                )

            if not id2label:
                id2label = _mapping_from_classes(
                    _find_value(
                        artifact,
                        [
                            "classes",
                            "class_names",
                            "labels",
                            "label_encoder",
                            "joint_label_encoder",
                        ],
                    )
                )

            if not id2label:
                raise KeyError(
                    "No class label mapping was found in the model checkpoint."
                )

            return {
                "model": model,
                "tokenizer": tokenizer,
                "id2label": id2label,
                "max_len": int(
                    _find_value(
                        artifact,
                        ["max_len", "max_length", "sequence_length"],
                        512,
                    )
                ),
            }

        # The common checkpoint format:
        # {
        #   "model_state_dict": ...,
        #   "model_name": ...,
        #   "id2label": ...,
        # }
        state_dict = _find_value(
            artifact,
            [
                "model_state_dict",
                "state_dict",
                "model_weights",
                "weights",
            ],
        )

        if state_dict is None:
            raise KeyError(
                "The checkpoint does not contain `model_state_dict`, "
                "`state_dict`, or a complete PyTorch model object."
            )

        model_name = _find_value(
            artifact,
            [
                "model_name",
                "base_model_name",
                "pretrained_model_name",
                "tokenizer_name",
            ],
        )

        if not model_name:
            raise KeyError(
                "The checkpoint does not contain the base model name. "
                "Save a field such as `model_name` when exporting the model."
            )

        id2label = _normalise_id2label(
            _find_value(artifact, ["id2label", "id_to_label"], {})
        )
        label2id = _normalise_label2id(
            _find_value(artifact, ["label2id", "label_to_id"], {})
        )

        if not id2label and label2id:
            id2label = {
                class_id: label
                for label, class_id in label2id.items()
            }

        if not id2label:
            id2label = _mapping_from_classes(
                _find_value(
                    artifact,
                    [
                        "classes",
                        "class_names",
                        "labels",
                        "label_encoder",
                        "joint_label_encoder",
                    ],
                )
            )

        if not id2label:
            raise KeyError(
                "The checkpoint does not contain its output class labels. "
                "Add `id2label`, `classes`, or a saved label encoder."
            )

        if not label2id:
            label2id = {
                label: class_id
                for class_id, label in id2label.items()
            }

        config = AutoConfig.from_pretrained(
            str(model_name),
            num_labels=len(id2label),
            id2label=id2label,
            label2id=label2id,
        )

        model = AutoModelForSequenceClassification.from_config(config)

        # Remove "module." prefixes created by DataParallel.
        if state_dict and all(
            str(key).startswith("module.")
            for key in state_dict.keys()
        ):
            state_dict = {
                str(key).removeprefix("module."): value
                for key, value in state_dict.items()
            }

        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as error:
            raise RuntimeError(
                "The checkpoint weights do not match "
                "AutoModelForSequenceClassification. The training notebook "
                "may have used a custom model class or a different number of "
                "labels. Save the full custom model object or export the model "
                "using save_pretrained(). "
                f"Original error: {error}"
            ) from error

        model.eval()
        tokenizer_name = _find_value(
            artifact,
            ["tokenizer_name", "model_name", "base_model_name"],
            model_name,
        )
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_name))

        return {
            "model": model,
            "tokenizer": tokenizer,
            "id2label": id2label,
            "max_len": int(
                _find_value(
                    artifact,
                    ["max_len", "max_length", "sequence_length"],
                    512,
                )
            ),
        }

    # --------------------------------------------------------
    # Case 2: the file directly contains a PyTorch model object
    # --------------------------------------------------------
    if isinstance(artifact, torch.nn.Module):
        model = artifact
        model.eval()

        config = getattr(model, "config", None)
        model_name = getattr(config, "_name_or_path", None)
        id2label = _normalise_id2label(
            getattr(config, "id2label", {})
        )

        if not model_name:
            raise KeyError(
                "The saved model object does not contain a base model name."
            )

        if not id2label:
            raise KeyError(
                "The saved model object does not contain id2label."
            )

        tokenizer = AutoTokenizer.from_pretrained(str(model_name))

        return {
            "model": model,
            "tokenizer": tokenizer,
            "id2label": id2label,
            "max_len": 512,
        }

    raise TypeError(
        "Unsupported model artifact. Expected a checkpoint dictionary "
        "or PyTorch model object."
    )


@lru_cache(maxsize=1)
def load_model() -> dict:
    """
    Download mbert_aspect_sentiment_model.pkl from Hugging Face once,
    load it, and cache it for the Streamlit session.
    """
    downloaded_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_MODEL_FILENAME,
        repo_type="model",
        token=os.getenv("HF_TOKEN") or None,
    )

    artifact = _load_artifact(downloaded_path)
    return _build_runtime(artifact)


# ============================================================
# TEXT PROCESSING AND INFERENCE
# ============================================================
def preprocess(text: str) -> str:
    """Clean a raw YouTube comment."""
    text = re.sub(r"http\S+|www\S+", "", str(text))
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^\w\s',.!?]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def _keyword_matches(text: str, keyword: str) -> bool:
    """Match a complete word or phrase."""
    pattern = rf"(?<!\w){re.escape(keyword.lower())}(?!\w)"
    return re.search(pattern, text.lower()) is not None


def extract_one_aspect(text: str) -> str:
    """
    Return exactly one aspect.

    This is used as a fallback when the trained output label does not include
    an aspect. The aspect with the highest keyword score is selected.
    """
    scores: dict[str, int] = {}

    for aspect in ASPECT_PRIORITY:
        matches = {
            keyword.lower()
            for keyword in ASPECTS[aspect]
            if _keyword_matches(text, keyword)
        }

        scores[aspect] = sum(
            2 if " " in keyword else 1
            for keyword in matches
        )

    best_aspect = max(
        ASPECT_PRIORITY,
        key=lambda aspect: (
            scores[aspect],
            -ASPECT_PRIORITY.index(aspect),
        ),
    )

    return best_aspect if scores[best_aspect] > 0 else "General"


def _canonical_sentiment(label: str) -> str | None:
    """Extract Positive, Neutral, or Negative from a trained class label."""
    value = str(label).strip().lower()

    if "positive" in value or re.search(r"(^|[^a-z])pos([^a-z]|$)", value):
        return "Positive"

    if "neutral" in value or re.search(r"(^|[^a-z])neu([^a-z]|$)", value):
        return "Neutral"

    if "negative" in value or re.search(r"(^|[^a-z])neg([^a-z]|$)", value):
        return "Negative"

    return None


def _aspect_from_joint_label(label: str) -> str | None:
    """Extract an aspect name from a joint aspect-sentiment class label."""
    cleaned = str(label).strip().lower()
    cleaned = re.sub(
        r"\b(positive|neutral|negative|pos|neu|neg)\b",
        " ",
        cleaned,
    )
    cleaned = re.sub(r"[_|:/\\\-]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # Prefer longer aliases first.
    for alias in sorted(ASPECT_ALIASES, key=len, reverse=True):
        if re.search(
            rf"(?<!\w){re.escape(alias)}(?!\w)",
            cleaned,
        ):
            return ASPECT_ALIASES[alias]

    return None


def predict_joint(
    clean_text: str,
    classifier: dict,
) -> tuple[str, str, float, str]:
    """
    Predict one trained output class and convert it into:
    sentiment, aspect, confidence, and raw model label.
    """
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
        output = model(**encoded)

        if not hasattr(output, "logits"):
            raise TypeError(
                "The loaded model output does not contain `logits`."
            )

        probabilities = torch.softmax(output.logits, dim=-1)[0]

    predicted_id = int(torch.argmax(probabilities).item())
    confidence = float(probabilities[predicted_id].item())
    raw_label = id2label.get(
        predicted_id,
        f"LABEL_{predicted_id}",
    )

    sentiment = _canonical_sentiment(raw_label)
    aspect = _aspect_from_joint_label(raw_label)

    # If the trained label only represents sentiment, use the one-aspect
    # keyword fallback so every comment still receives exactly one aspect.
    if aspect is None:
        aspect = extract_one_aspect(clean_text)

    if sentiment is None:
        raise ValueError(
            "The predicted class label does not contain Positive, Neutral, "
            f"or Negative. Predicted raw label: {raw_label!r}. "
            "Check the id2label/classes saved inside the checkpoint."
        )

    return sentiment, aspect, round(confidence, 4), str(raw_label)


def analyse_comment(text: str, classifier: dict) -> dict:
    """
    Full application flow:
    preprocess -> trained mBERT -> one sentiment -> one aspect.
    """
    clean = preprocess(text)

    if not clean:
        raise ValueError("The comment is empty after preprocessing.")

    sentiment, aspect, confidence, raw_label = predict_joint(
        clean,
        classifier,
    )

    return {
        "original": text,
        "clean": clean,
        "sentiment": sentiment,
        "confidence": confidence,

        # Keep a one-item list because the existing app.py uses:
        # for aspect in result["aspects"]
        "aspects": [aspect],

        # Useful for checking what the trained model actually predicted.
        "raw_label": raw_label,
    }
