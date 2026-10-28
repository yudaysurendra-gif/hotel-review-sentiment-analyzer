"""Shared helpers: text cleaning, label mapping, aspect detection, prediction."""
import re
import joblib
import pandas as pd

MODEL_PATH = "models/sentiment_model.joblib"

ASPECTS = {
    "Cleanliness": ["clean", "dirty", "spotless", "filthy", "smell", "bed bugs", "hygiene"],
    "Staff & Service": ["staff", "service", "reception", "friendly", "rude", "helpful", "check-in"],
    "Location": ["location", "beach", "near", "walking distance", "noisy", "quiet", "central"],
    "Food": ["food", "breakfast", "buffet", "restaurant", "dinner", "tasty", "delicious"],
    "Room & Comfort": ["room", "bed", "bathroom", "pillow", "balcony", "view", "air conditioning"],
    "Value for Money": ["price", "value", "expensive", "overpriced", "cheap", "worth", "money"],
}


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z\s'-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def rating_to_sentiment(rating: float) -> str:
    """1-2 stars = Negative, 3 = Neutral, 4-5 = Positive."""
    if rating <= 2:
        return "Negative"
    if rating == 3:
        return "Neutral"
    return "Positive"


def load_model(path: str = MODEL_PATH):
    return joblib.load(path)


def predict(model, reviews):
    """Return a DataFrame with sentiment and confidence for each review."""
    cleaned = [clean_text(r) for r in reviews]
    probs = model.predict_proba(cleaned)
    labels = model.classes_[probs.argmax(axis=1)]
    return pd.DataFrame({
        "Review": list(reviews),
        "Sentiment": labels,
        "Confidence": (probs.max(axis=1) * 100).round(1),
    })


def detect_aspects(review: str):
    """Find which hotel aspects a review talks about (simple keyword match)."""
    text = clean_text(review)
    return [a for a, kws in ASPECTS.items() if any(k in text for k in kws)]
