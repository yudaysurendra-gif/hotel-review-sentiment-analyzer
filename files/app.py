"""Streamlit web app for the Hotel Review Sentiment Analyzer.
Run with: streamlit run app.py"""
import os
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from analyzer import MODEL_PATH, detect_aspects, load_model, predict

st.set_page_config(page_title="Hotel Review Sentiment Analyzer", page_icon="🏨", layout="wide")
st.title("🏨 Hotel Review Sentiment Analyzer")
st.caption("Classify hotel reviews as Positive, Neutral or Negative and see which aspects guests talk about.")

if not os.path.exists(MODEL_PATH):
    st.error("Model not found. Run `python train_model.py` first.")
    st.stop()


@st.cache_resource
def get_model():
    return load_model()


model = get_model()
COLORS = {"Positive": "#2e9e5b", "Neutral": "#e0a800", "Negative": "#d64545"}
tab1, tab2 = st.tabs(["Single Review", "Batch Analysis (CSV)"])

with tab1:
    review = st.text_area("Paste a hotel review", height=150,
                          placeholder="The staff were friendly and the room was spotless...")
    if st.button("Analyze", type="primary") and review.strip():
        result = predict(model, [review]).iloc[0]
        c1, c2 = st.columns(2)
        c1.metric("Sentiment", result["Sentiment"])
        c2.metric("Confidence", f"{result['Confidence']}%")
        aspects = detect_aspects(review)
        st.write("**Aspects mentioned:**", ", ".join(aspects) if aspects else "None detected")

with tab2:
    st.write("Upload a CSV with a **Review** column (a Rating column is optional).")
    file = st.file_uploader("Upload CSV", type="csv")
    if file:
        data = pd.read_csv(file)
        if "Review" not in data.columns:
            st.error("The CSV must contain a 'Review' column.")
        else:
            data = data.dropna(subset=["Review"]).reset_index(drop=True)
            out = predict(model, data["Review"].tolist())
            out["Aspects"] = out["Review"].apply(lambda r: ", ".join(detect_aspects(r)))

            counts = out["Sentiment"].value_counts()
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total reviews", len(out))
            for col, label in zip((m2, m3, m4), ("Positive", "Neutral", "Negative")):
                col.metric(label, int(counts.get(label, 0)))

            left, right = st.columns(2)
            with left:
                st.subheader("Sentiment distribution")
                fig, ax = plt.subplots()
                ax.bar(counts.index, counts.values, color=[COLORS[s] for s in counts.index])
                ax.set_ylabel("Reviews")
                st.pyplot(fig)
            with right:
                st.subheader("Sentiment by aspect")
                rows = []
                for _, r in out.iterrows():
                    for a in filter(None, r["Aspects"].split(", ")):
                        rows.append((a, r["Sentiment"]))
                if rows:
                    pivot = (pd.DataFrame(rows, columns=["Aspect", "Sentiment"])
                             .groupby(["Aspect", "Sentiment"]).size().unstack(fill_value=0))
                    fig2, ax2 = plt.subplots()
                    pivot.plot(kind="barh", stacked=True, ax=ax2,
                               color=[COLORS[c] for c in pivot.columns])
                    st.pyplot(fig2)

            st.subheader("Results")
            st.dataframe(out, use_container_width=True)
            st.download_button("Download results CSV", out.to_csv(index=False).encode(),
                               "sentiment_results.csv", "text/csv")
