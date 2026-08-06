"""
app.py
Streamlit demo app for the Explainable Fake News Detector.

Run with:
    streamlit run app.py

What it does:
  1. Loads the trained pipeline (TF-IDF + Logistic Regression)
  2. Takes a news article/headline as input
  3. Predicts FAKE or REAL with a confidence score
  4. Uses LIME to explain WHICH words pushed the prediction
     toward FAKE vs REAL, both as a highlighted text view
     and a bar chart.
"""

import json
import os

import joblib
import matplotlib.pyplot as plt
import streamlit as st
from lime.lime_text import LimeTextExplainer

from preprocess import clean_text

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODELS_DIR, "fake_news_pipeline.joblib")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")


# ---------- Caching so the model only loads once ----------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    if not os.path.exists(METRICS_PATH):
        return None
    with open(METRICS_PATH) as f:
        return json.load(f)


def predict_proba_wrapper(model):
    """
    LIME needs a function: list[str] -> np.array of class probabilities.
    Our sklearn Pipeline already handles raw text -> vectorize -> predict_proba,
    but we clean the text first so it matches how the model was trained.
    """
    def _predict(texts):
        cleaned = [clean_text(t) for t in texts]
        return model.predict_proba(cleaned)
    return _predict


def highlight_text(text: str, word_weights: dict) -> str:
    """
    Builds an HTML string where words are highlighted:
    green = pushes toward REAL, red = pushes toward FAKE.
    Opacity scales with the magnitude of the LIME weight.
    """
    max_w = max([abs(w) for w in word_weights.values()], default=1e-6)
    html_words = []
    for raw_word in text.split():
        key = raw_word.lower().strip(".,!?\"'()")
        weight = word_weights.get(key)
        if weight is None:
            html_words.append(raw_word)
            continue
        intensity = min(abs(weight) / max_w, 1.0)
        color = f"rgba(220,50,50,{intensity:.2f})" if weight < 0 else f"rgba(40,160,80,{intensity:.2f})"
        html_words.append(
            f'<span style="background-color:{color}; padding:1px 2px; border-radius:3px;">{raw_word}</span>'
        )
    return " ".join(html_words)


def main():
    st.set_page_config(page_title="Explainable Fake News Detector", page_icon="📰", layout="wide")
    st.title("📰 Explainable Fake News Detector")
    st.caption("Predicts whether a news article is FAKE or REAL, and explains which words drove the decision (via LIME).")

    model = load_model()
    metrics = load_metrics()

    if model is None:
        st.error(
            "No trained model found at `models/fake_news_pipeline.joblib`.\n\n"
            "Train one first, e.g.:\n\n"
            "```\npython data/make_sample_data.py\n"
            "python train_model.py --data data/sample_news.csv --text_col text --label_col label\n```"
        )
        st.stop()

    with st.sidebar:
        st.header("Model info")
        if metrics:
            st.metric("Accuracy", f"{metrics['accuracy']*100:.1f}%")
            st.metric("F1-score", f"{metrics['f1']*100:.1f}%")
            st.caption(f"Trained on {metrics['n_train']} examples, tested on {metrics['n_test']}.")
        else:
            st.caption("No metrics.json found -- run train_model.py to generate it.")
        st.divider()
        st.header("How explanations work")
        st.write(
            "LIME slightly perturbs the input text (removing words) and observes "
            "how the prediction changes, to estimate which words matter most. "
            "🟩 Green = pushes toward REAL. 🟥 Red = pushes toward FAKE."
        )

    classes = list(model.classes_)
    explainer = LimeTextExplainer(class_names=classes)

    default_text = (
        "Scientists confirm the earth will stop spinning next month, "
        "government refuses to warn the public about the disaster!"
    )
    user_text = st.text_area("Paste a news headline or article:", value=default_text, height=150)
    num_features = st.slider("Number of words to explain", min_value=4, max_value=20, value=8)

    if st.button("Analyze", type="primary") and user_text.strip():
        with st.spinner("Predicting and generating explanation..."):
            cleaned = clean_text(user_text)
            probs = model.predict_proba([cleaned])[0]
            pred_idx = probs.argmax()
            pred_label = classes[pred_idx]
            confidence = probs[pred_idx]

            explanation = explainer.explain_instance(
                cleaned,
                predict_proba_wrapper(model),
                num_features=num_features,
                labels=(pred_idx,),
            )
            word_weights = dict(explanation.as_list(label=pred_idx))

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Prediction")
            if pred_label.upper() == "FAKE":
                st.error(f"🚨 Predicted: **{pred_label}** ({confidence*100:.1f}% confidence)")
            else:
                st.success(f"✅ Predicted: **{pred_label}** ({confidence*100:.1f}% confidence)")

            st.write("Class probabilities:")
            for cls, p in zip(classes, probs):
                st.progress(float(p), text=f"{cls}: {p*100:.1f}%")

            st.subheader("Highlighted explanation")
            st.markdown(
                f'<div style="line-height:2.0; font-size:16px;">{highlight_text(user_text, word_weights)}</div>',
                unsafe_allow_html=True,
            )

        with col2:
            st.subheader("Top contributing words")
            words = [w for w, _ in explanation.as_list(label=pred_idx)]
            weights = [wt for _, wt in explanation.as_list(label=pred_idx)]
            colors = ["#d63232" if w < 0 else "#28a050" for w in weights]

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.barh(words, weights, color=colors)
            ax.set_xlabel("Contribution to prediction")
            ax.axvline(0, color="black", linewidth=0.8)
            ax.invert_yaxis()
            st.pyplot(fig)

            st.caption(
                "Bars pointing right (green) push toward REAL. "
                "Bars pointing left (red) push toward FAKE."
            )

    st.divider()
    st.caption(
        "⚠️ This is a student/demo project, not a fact-checking authority. "
        "Predictions are based on writing style/word patterns learned from the training data, "
        "not verified facts."
    )


if __name__ == "__main__":
    main()
