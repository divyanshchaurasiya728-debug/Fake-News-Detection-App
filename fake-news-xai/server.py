"""
server.py
Flask backend for the Explainable Fake News Detector -- serves the
"TruthLens" custom HTML/CSS/JS frontend (in frontend/) and exposes the
real trained model + LIME explanations + transcription/report generation
as a JSON API for that frontend's script.js to call.

Run with:
    python server.py
Then open http://localhost:5000 in a browser.

Pages (served as static files, matching the frontend's own relative links):
    GET  /                      -> frontend/index.html   (login page)
    GET  /dashboard.html        -> frontend/dashboard.html
    GET  /history.html          -> frontend/history.html
    GET  /about.html            -> frontend/about.html
    GET  /settings.html         -> frontend/settings.html

API endpoints (called by frontend/script.js):
    GET  /api/metrics           -> model accuracy/F1/train-test size
    POST /api/analyze           -> {text} -> prediction + LIME explanation
    POST /api/fetch-url         -> {url} -> {title, text}  (not yet wired into this frontend's UI)
    POST /api/transcribe        -> multipart file upload (audio or video) -> {text}
    POST /api/report            -> analysis result -> PDF file download (not yet wired into this frontend's UI)
"""

import json
import os
import tempfile
from io import BytesIO

import joblib
import requests
from flask import Flask, jsonify, request, send_file, send_from_directory
from lime.lime_text import LimeTextExplainer

from preprocess import clean_text
from url_utils import fetch_article_from_url
from media_utils import transcribe_media_file
from report_utils import generate_pdf_report

BASE_DIR = os.path.dirname(__file__)
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "fake_news_pipeline.joblib")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")

# static_folder points at the frontend directory and static_url_path=""
# means its files are served straight from "/", so the frontend's own
# relative links (href="style.css", src="script.js", href="dashboard.html")
# work completely unchanged -- no template rewriting needed.
app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")

_model = None
_explainer = None


def get_model():
    global _model, _explainer
    if _model is None and os.path.exists(MODEL_PATH):
        _model = joblib.load(MODEL_PATH)
        _explainer = LimeTextExplainer(class_names=list(_model.classes_))
    return _model, _explainer


def predict_proba_wrapper(model):
    def _predict(texts):
        cleaned = [clean_text(t) for t in texts]
        return model.predict_proba(cleaned)
    return _predict


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/metrics")
def api_metrics():
    if not os.path.exists(METRICS_PATH):
        return jsonify({"available": False})
    with open(METRICS_PATH) as f:
        metrics = json.load(f)
    metrics["available"] = True
    return jsonify(metrics)


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    model, explainer = get_model()
    if model is None:
        return jsonify({"error": "No trained model found. Train one with train_model.py first."}), 503

    data = request.get_json(force=True)
    text = (data or {}).get("text", "").strip()
    num_features = int((data or {}).get("num_features", 8))

    if not text:
        return jsonify({"error": "No text provided."}), 400

    cleaned = clean_text(text)
    classes = list(model.classes_)
    probs = model.predict_proba([cleaned])[0]
    pred_idx = int(probs.argmax())
    pred_label = classes[pred_idx]
    confidence = float(probs[pred_idx])

    explanation = explainer.explain_instance(
        cleaned,
        predict_proba_wrapper(model),
        num_features=num_features,
        labels=(pred_idx,),
    )
    word_weights = explanation.as_list(label=pred_idx)

    return jsonify({
        "label": pred_label,
        "confidence": confidence,
        "probs": {cls: float(p) for cls, p in zip(classes, probs)},
        "word_weights": word_weights,  # [[word, weight], ...]
        "input_text": text,
    })


@app.route("/api/fetch-url", methods=["POST"])
def api_fetch_url():
    data = request.get_json(force=True)
    url = (data or {}).get("url", "").strip()
    if not url:
        return jsonify({"error": "No URL provided."}), 400
    try:
        result = fetch_article_from_url(url)
        return jsonify(result)
    except requests.exceptions.ConnectionError:
        return jsonify({"error": "Couldn't reach that URL. Check the address and your internet connection."}), 422
    except requests.exceptions.Timeout:
        return jsonify({"error": "The request timed out. The site may be slow or unreachable."}), 422
    except requests.exceptions.HTTPError as e:
        return jsonify({"error": f"The site returned an error ({e.response.status_code})."}), 422
    except ValueError as e:
        return jsonify({"error": str(e)}), 422
    except Exception:
        return jsonify({"error": "Something went wrong fetching that article."}), 422


@app.route("/api/transcribe", methods=["POST"])
def api_transcribe():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400
    file = request.files["file"]
    suffix = os.path.splitext(file.filename)[1] or ".bin"

    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    file.save(tmp_path)

    try:
        transcript = transcribe_media_file(tmp_path)
        return jsonify({"text": transcript})
    except Exception as e:
        return jsonify({"error": str(e)}), 422
    finally:
        os.remove(tmp_path)


@app.route("/api/report", methods=["POST"])
def api_report():
    data = request.get_json(force=True)
    required = ["input_text", "label", "confidence", "probs", "word_weights"]
    if not data or any(k not in data for k in required):
        return jsonify({"error": "Missing fields for report generation."}), 400

    pdf_bytes = generate_pdf_report(
        article_text=data["input_text"],
        label=data["label"],
        confidence=float(data["confidence"]),
        class_probs=data["probs"],
        word_weights=[tuple(wp) for wp in data["word_weights"]],
        source=data.get("source", "Pasted text"),
    )
    return send_file(
        BytesIO(pdf_bytes),
        mimetype="application/pdf",
        as_attachment=True,
        download_name="fake_news_report.pdf",
    )


if __name__ == "__main__":
    app.run(debug=False, port=5000)
