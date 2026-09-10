"""
app.py
Streamlit demo app for the Explainable Fake News Detector -- polished UI version
with multiple input modes (text, URL, audio, video) and a downloadable PDF report.

Run with:
    streamlit run app.py

What it does:
  1. Loads the trained pipeline (TF-IDF + Logistic Regression)
  2. Accepts input as: pasted text, a news article URL, an audio clip, or a video file
  3. Predicts FAKE or REAL with a confidence score
  4. Uses LIME to explain WHICH words pushed the prediction
     toward FAKE vs REAL, both as a highlighted text view and a bar chart
  5. Lets you download a PDF report of the result
"""

import json
import os
import tempfile

import joblib
import matplotlib.pyplot as plt
import streamlit as st
from lime.lime_text import LimeTextExplainer

from preprocess import clean_text
from url_utils import fetch_article_from_url
from media_utils import transcribe_media_file
from report_utils import generate_pdf_report

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODELS_DIR, "fake_news_pipeline.joblib")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")

EXAMPLES = [
    "Scientists confirm the earth will stop spinning next month, government refuses to warn the public about the disaster!",
    "The central bank raised interest rates by a quarter point on Wednesday, citing persistent inflation pressures.",
    "Doctors HATE this one weird trick that cures all diseases instantly, Big Pharma doesn't want you to know!",
    "The city council approved the annual budget after months of public hearings and revisions to the plan.",
]

CUSTOM_CSS = """
<style>
    .block-container { padding-top: 2.2rem; max-width: 1100px; }

    .hero {
        text-align: center;
        padding: 0.5rem 0 1.6rem 0;
    }
    .hero h1 {
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
        background: linear-gradient(90deg, #6C5CE7, #00CEC9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero p {
        color: #A0A4B8;
        font-size: 1.02rem;
        margin-top: 0;
    }

    .stTextArea textarea {
        border-radius: 12px !important;
        border: 1px solid #2A2F3E !important;
        font-size: 1.02rem !important;
    }

    div.stButton > button {
        border-radius: 10px;
        font-weight: 600;
        padding: 0.55rem 1.4rem;
        border: none;
    }
    div.stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #6C5CE7, #8154F2);
    }
    div.stDownloadButton > button {
        border-radius: 10px;
        font-weight: 600;
        border: 1px solid #6C5CE7;
    }

    .example-chip button {
        background-color: #1B1F2A !important;
        color: #C7CAD9 !important;
        font-size: 0.82rem !important;
        font-weight: 400 !important;
        border: 1px solid #2A2F3E !important;
        border-radius: 20px !important;
        padding: 0.3rem 0.9rem !important;
    }

    .result-card {
        border-radius: 14px;
        padding: 1.3rem 1.5rem;
        margin-bottom: 1rem;
        font-size: 1.15rem;
        font-weight: 700;
    }
    .result-fake {
        background: rgba(220, 50, 50, 0.12);
        border: 1px solid rgba(220, 50, 50, 0.4);
        color: #FF6B6B;
    }
    .result-real {
        background: rgba(40, 160, 80, 0.12);
        border: 1px solid rgba(40, 160, 80, 0.4);
        color: #4ADE80;
    }

    .metric-box {
        background-color: #161A23;
        border: 1px solid #2A2F3E;
        border-radius: 12px;
        padding: 0.9rem 1rem;
        text-align: center;
    }
    .metric-box .label { color: #A0A4B8; font-size: 0.8rem; }
    .metric-box .value { font-size: 1.6rem; font-weight: 800; color: #F5F6FA; }

    .highlight-box {
        background-color: #161A23;
        border: 1px solid #2A2F3E;
        border-radius: 12px;
        padding: 1.1rem 1.3rem;
        line-height: 2.1;
        font-size: 1.02rem;
    }

    .footer-note {
        text-align: center;
        color: #6B6F80;
        font-size: 0.8rem;
        margin-top: 2rem;
    }
</style>
"""


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
    We clean the text first so it matches how the model was trained.
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
        color = f"rgba(255,107,107,{intensity:.2f})" if weight < 0 else f"rgba(74,222,128,{intensity:.2f})"
        html_words.append(
            f'<span style="background-color:{color}; padding:1px 4px; border-radius:5px;">{raw_word}</span>'
        )
    return " ".join(html_words)


def render_input_tabs():
    """
    Renders the Text / URL / Audio / Video input tabs.
    Whichever one the user submits sets st.session_state.article_text
    (and .source_label, used later in the PDF report).
    """
    if "article_text" not in st.session_state:
        st.session_state.article_text = EXAMPLES[0]
    if "source_label" not in st.session_state:
        st.session_state.source_label = "Pasted text"

    tab_text, tab_url, tab_audio, tab_video = st.tabs(["📝 Text", "🔗 URL", "🎵 Audio", "🎬 Video"])

    with tab_text:
        st.markdown("**Try an example:**")
        ex_cols = st.columns(len(EXAMPLES))
        for i, ex in enumerate(EXAMPLES):
            with ex_cols[i]:
                st.markdown('<div class="example-chip">', unsafe_allow_html=True)
                label = (ex[:34] + "…") if len(ex) > 34 else ex
                if st.button(label, key=f"ex_{i}", help=ex):
                    st.session_state.article_text = ex
                    st.session_state.source_label = "Pasted text (example)"
                st.markdown("</div>", unsafe_allow_html=True)

        st.text_area(
            "Paste a news headline or article:",
            key="article_text",
            height=140,
        )

    with tab_url:
        url = st.text_input("News article URL:", placeholder="https://example.com/some-article")
        if st.button("Fetch article", key="fetch_url_btn"):
            if not url.strip():
                st.warning("Please enter a URL first.")
            else:
                with st.spinner("Fetching and extracting article text..."):
                    try:
                        result = fetch_article_from_url(url.strip())
                        st.session_state.article_text = result["text"]
                        st.session_state.source_label = f"URL: {url.strip()}"
                        st.success(f"Fetched \"{result['title']}\" -- {len(result['text'].split())} words extracted.")
                    except Exception as e:
                        st.error(f"Couldn't extract article text: {e}")

    with tab_audio:
        audio_file = st.file_uploader("Upload an audio clip", type=["wav", "mp3", "m4a", "ogg", "flac"])
        if audio_file and st.button("Transcribe audio", key="transcribe_audio_btn"):
            with st.spinner("Transcribing audio (this can take a moment)..."):
                suffix = os.path.splitext(audio_file.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(audio_file.read())
                    tmp_path = tmp.name
                try:
                    transcript = transcribe_media_file(tmp_path)
                    st.session_state.article_text = transcript
                    st.session_state.source_label = f"Audio: {audio_file.name}"
                    st.success(f"Transcribed {len(transcript.split())} words.")
                except Exception as e:
                    st.error(f"Transcription failed: {e}")
                finally:
                    os.remove(tmp_path)

    with tab_video:
        video_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "mkv", "avi", "webm"])
        if video_file and st.button("Transcribe video", key="transcribe_video_btn"):
            with st.spinner("Extracting audio and transcribing (this can take a moment)..."):
                suffix = os.path.splitext(video_file.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(video_file.read())
                    tmp_path = tmp.name
                try:
                    transcript = transcribe_media_file(tmp_path)
                    st.session_state.article_text = transcript
                    st.session_state.source_label = f"Video: {video_file.name}"
                    st.success(f"Transcribed {len(transcript.split())} words.")
                except Exception as e:
                    st.error(f"Transcription failed: {e}")
                finally:
                    os.remove(tmp_path)

    if st.session_state.source_label != "Pasted text":
        st.caption(f"Current input source: **{st.session_state.source_label}**")
        with st.expander("Preview extracted/transcribed text"):
            st.write(st.session_state.article_text)


def main():
    st.set_page_config(page_title="Explainable Fake News Detector", page_icon="📰", layout="wide")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    model = load_model()
    metrics = load_metrics()

    st.markdown(
        """
        <div class="hero">
            <h1>📰 Explainable Fake News Detector</h1>
            <p>Predicts FAKE vs REAL &mdash; from text, a URL, audio, or video &mdash; and shows exactly which words drove the decision.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if model is None:
        st.error(
            "No trained model found at `models/fake_news_pipeline.joblib`.\n\n"
            "Train one first, e.g.:\n\n"
            "```\npython data/make_sample_data.py\n"
            "python train_model.py --data data/sample_news.csv --text_col text --label_col label\n```"
        )
        st.stop()

    # ---------- Model info strip ----------
    if metrics:
        c1, c2, c3, c4 = st.columns(4)
        for col, label, value in [
            (c1, "Accuracy", f"{metrics['accuracy']*100:.1f}%"),
            (c2, "F1-score", f"{metrics['f1']*100:.1f}%"),
            (c3, "Train size", f"{metrics['n_train']:,}"),
            (c4, "Test size", f"{metrics['n_test']:,}"),
        ]:
            with col:
                st.markdown(
                    f'<div class="metric-box"><div class="value">{value}</div>'
                    f'<div class="label">{label}</div></div>',
                    unsafe_allow_html=True,
                )
        st.write("")

    classes = list(model.classes_)
    explainer = LimeTextExplainer(class_names=classes)

    render_input_tabs()
    user_text = st.session_state.article_text

    col_a, col_b = st.columns([3, 1])
    with col_b:
        num_features = st.slider("Words to explain", min_value=4, max_value=20, value=8)
    with col_a:
        st.write("")
        analyze_clicked = st.button("🔍 Analyze", type="primary", use_container_width=False)

    if analyze_clicked and user_text.strip():
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
            word_weights_list = explanation.as_list(label=pred_idx)
            word_weights = dict(word_weights_list)

        st.write("")
        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            css_class = "result-fake" if pred_label.upper() == "FAKE" else "result-real"
            icon = "🚨" if pred_label.upper() == "FAKE" else "✅"
            st.markdown(
                f'<div class="result-card {css_class}">{icon} Predicted: {pred_label} '
                f'&nbsp;·&nbsp; {confidence*100:.1f}% confidence</div>',
                unsafe_allow_html=True,
            )

            for cls, p in zip(classes, probs):
                st.write(f"**{cls}**")
                st.progress(float(p), text=f"{p*100:.1f}%")

            st.write("")
            st.markdown("**Highlighted explanation**")
            st.markdown(
                f'<div class="highlight-box">{highlight_text(user_text, word_weights)}</div>',
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown("**Top contributing words**")
            words = [w for w, _ in word_weights_list]
            weights = [wt for _, wt in word_weights_list]
            colors = ["#FF6B6B" if w < 0 else "#4ADE80" for w in weights]

            plt.style.use("dark_background")
            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_facecolor("#0E1117")
            ax.set_facecolor("#0E1117")
            ax.barh(words, weights, color=colors)
            ax.set_xlabel("Contribution to prediction")
            ax.axvline(0, color="#6B6F80", linewidth=0.8)
            ax.invert_yaxis()
            for spine in ax.spines.values():
                spine.set_color("#2A2F3E")
            st.pyplot(fig)

            st.caption("🟢 Green pushes toward REAL &nbsp;&nbsp; 🔴 Red pushes toward FAKE")

        st.write("")
        class_probs = {cls: float(p) for cls, p in zip(classes, probs)}
        pdf_bytes = generate_pdf_report(
            article_text=user_text,
            label=pred_label,
            confidence=float(confidence),
            class_probs=class_probs,
            word_weights=word_weights_list,
            source=st.session_state.source_label,
        )
        st.download_button(
            "⬇️ Download PDF report",
            data=pdf_bytes,
            file_name="fake_news_report.pdf",
            mime="application/pdf",
        )

    st.markdown(
        '<div class="footer-note">⚠️ Student/demo project, not a fact-checking authority. '
        "Predictions reflect writing-style patterns learned from training data, not verified facts.</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
