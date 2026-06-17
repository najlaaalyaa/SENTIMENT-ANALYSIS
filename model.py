"""
model.py — mBERT sentiment classification + aspect-based extraction
Used by app.py via import
"""

import re
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# ── Cooking aspect keyword dictionary (Malay) ───────
ASPECTS = {
    "Taste / Rasa": [
        "sedap","lazat","lezat","rasa","masin","manis","masam","pahit","pedas",
        "tawar","enak","nyaman","lemak","taste","delicious","flavour","flavor",
        "yummy","tasteless","bland","sweet","salty","sour","spicy","bitter",
        "kurang sedap","tak sedap","sangat sedap","memang sedap","sedap sangat"
    ],
    "Ingredients / Bahan": [
        "bahan","sukatan","resepi","resipi","ramuan","ganti","kurang","lebih",
        "ingredient","recipe","substitute","measurement","quantity","amount",
        "portion","spice","rempah","santan","minyak","garam","gula","tepung",
        "sayur","daging","ayam","ikan","udang","telur",
    ],
    "Cooking Steps / Langkah": [
        "cara","langkah","kaedah","teknik","proses","mudah","susah","sukar",
        "method","step","process","easy","hard","difficult","simple","follow",
        "instructions","tutorial","guide","demo","ikut","faham","jelas",
        "peringkat","prosedur","arahan",
    ],
    "General": [
        "lama","cepat","lambat","minit","jam","masa","duration","quick",
        "slow","fast","long","short","minute","hour","time","tempoh",
        "sekejap","sebentar","berapa lama",
        "cantik","comel","menarik","kemas","presentation","plating","look",
        "beautiful","nice","neat","video","quality","visual","gambar","foto",
        "warna","colour","color","hiasan","garnish",
        "lembut","keras","rangup","gebu","moist","crispy","crunchy","soft",
        "hard","fluffy","dry","wet","texture","tekstur","kenyal","garing",
        "berderai","halus","kasar","licin",
    ],
}


def load_model():
    """Load mBERT multilingual sentiment model from HuggingFace."""
    MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"
    tokenizer  = AutoTokenizer.from_pretrained("qiyuw/WSPAlign-mbert-base")
    model      = AutoModelForSequenceClassification.from_pretrained("qiyuw/WSPAlign-mbert-base")
    classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=-1,       # CPU; change to 0 for GPU
        truncation=True,
        max_length=512,
    )
    return classifier


def preprocess(text: str) -> str:
    """Clean raw YouTube comment text."""
    text = re.sub(r"http\S+|www\S+", "", text)          # remove URLs
    text = re.sub(r"@\w+", "", text)                     # remove @mentions
    text = re.sub(r"[^\w\s',.!?]", " ", text)            # remove special chars / emojis
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def map_label(label: str, score: float) -> tuple[str, float]:
    """
    Convert mBERT 1-5 star output to Positive / Neutral / Negative.
    Returns (sentiment_label, confidence_0_to_1).
    """
    star = int(label.split()[0])
    if star >= 4:
        sentiment = "Positive"
    elif star == 3:
        sentiment = "Neutral"
    else:
        sentiment = "Negative"
    return sentiment, round(score, 4)


def extract_aspects(text: str) -> list:
    """Keyword-based aspect extraction for Malay cooking comments."""
    text_lower = text.lower()
    found = [aspect for aspect, keywords in ASPECTS.items()
             if any(kw in text_lower for kw in keywords)]
    return found if found else ["General"]


def analyse(text: str, classifier) -> dict:
    """
    Full pipeline: preprocess → mBERT inference → aspect extraction.

    Returns:
        {
          original: str,
          clean: str,
          sentiment: "Positive" | "Neutral" | "Negative",
          confidence: float,
          aspects: list[str],
        }
    """
    clean  = preprocess(text)
    result = classifier(clean[:512])[0]
    sentiment, confidence = map_label(result["label"], result["score"])
    aspects = extract_aspects(clean)
    return {
        "original":   text,
        "clean":      clean,
        "sentiment":  sentiment,
        "confidence": confidence,
        "aspects":    aspects,
    }
