"""Train and evaluate the sentiment model.
Usage: python train_model.py [path/to/reviews.csv]
CSV needs columns: Review, Rating (1-5)."""
import sys
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from analyzer import MODEL_PATH, clean_text, rating_to_sentiment

csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/hotel_reviews.csv"
df = pd.read_csv(csv_path).dropna(subset=["Review", "Rating"])
df["clean"] = df["Review"].apply(clean_text)
df["Sentiment"] = df["Rating"].apply(rating_to_sentiment)

print("Class distribution:\n", df["Sentiment"].value_counts(), "\n")

X_train, X_test, y_train, y_test = train_test_split(
    df["clean"], df["Sentiment"], test_size=0.2, random_state=42, stratify=df["Sentiment"]
)

model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, stop_words="english", sublinear_tf=True)),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
])
model.fit(X_train, y_train)

pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, pred):.3f}\n")
print(classification_report(y_test, pred))
print("Confusion matrix (rows = actual, cols = predicted):")
print(model.classes_)
print(confusion_matrix(y_test, pred, labels=model.classes_))

joblib.dump(model, MODEL_PATH)
print(f"\nModel saved to {MODEL_PATH}")
