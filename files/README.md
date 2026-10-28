# Hotel Review Sentiment Analyzer

Classifies hotel reviews as **Positive / Neutral / Negative** using TF-IDF + Logistic Regression,
detects which aspects (cleanliness, staff, food, location, room, value) guests mention,
and presents everything in a Streamlit dashboard.

## Setup
```bash
pip install -r requirements.txt
python generate_sample_data.py     # creates data/hotel_reviews.csv (synthetic)
python train_model.py              # trains and saves models/sentiment_model.joblib
streamlit run app.py               # launches the web app
```

## Use real data
Download a dataset such as Kaggle's *TripAdvisor Hotel Reviews* (columns `Review`, `Rating`) and run:
```bash
python train_model.py path/to/tripadvisor_hotel_reviews.csv
```

## Structure
- `analyzer.py` – cleaning, label mapping, aspect detection, prediction
- `train_model.py` – training and evaluation
- `app.py` – Streamlit UI (single review + batch CSV)
- `generate_sample_data.py` – synthetic starter data

## Ideas to extend
- Swap in a transformer model (DistilBERT) for higher accuracy
- Aspect-level sentiment instead of keyword spotting
- Deploy on Azure App Service and store results in Azure SQL
