# Explainable AI for Fake News Detection (TruthLens)

A fake news detector that predicts whether a piece of news looks **FAKE** or **REAL** and shows **which words influenced the decision**. It uses a scikit-learn classifier (TF-IDF + Logistic Regression) with LIME for word-level explanations, a Flask backend, and a custom web interface called **TruthLens**. It accepts pasted text, article links, audio clips and video files.

## Features

- **Classification** with a confidence score and a FAKE / REAL / Uncertain status
- **Explainability** with LIME, which scores each word by how much it pushed the prediction
- **Four input types**: text, article URL, audio and video (speech is transcribed to text first)
- **TruthLens web app** with login, dashboard, analysis report, history, about and settings pages
- **PDF reports** of an analysis
- **Streamlit version** of the app, with LIME word highlighting and a bar chart

## Two interfaces

| Interface | Start with | Opens at | What it shows |
|---|---|---|---|
| TruthLens (Flask) | `python server.py` | http://localhost:5000 | Fake probability, status, key findings, saved history |
| Streamlit | `streamlit run app.py` | http://localhost:8501 | Verdict, LIME word highlighting, word bar chart, PDF download |

Both use the same trained model in `models/`.

## Project structure

```
Fake-News-Detection-App/
└── fake-news-xai/
    ├── server.py            # Flask backend: serves TruthLens and the JSON API
    ├── app.py               # Streamlit app
    ├── preprocess.py        # text cleaning shared by training and the apps
    ├── train_model.py       # trains and saves the model
    ├── tune_model.py        # optional grid-search tuning (slower, usually better)
    ├── url_utils.py         # extracts article text from a URL
    ├── media_utils.py       # transcribes audio and video to text
    ├── report_utils.py      # builds the PDF report
    ├── requirements.txt
    ├── packages.txt         # system packages for deployment (ffmpeg)
    ├── .streamlit/
    │   └── config.toml      # Streamlit theme
    ├── frontend/            # TruthLens pages
    │   ├── index.html       # login
    │   ├── dashboard.html
    │   ├── history.html
    │   ├── about.html
    │   ├── settings.html
    │   ├── style.css
    │   └── script.js
    ├── data/                # datasets (not committed)
    └── models/              # trained model and metrics
```

## Setup

**Requirements:** Python 3.11 or newer (works on 3.14 with scikit-learn 1.8 or newer) and **ffmpeg** for audio and video.

1. Clone and enter the project:

   ```bash
   git clone https://github.com/divyanshchaurasiya728-debug/Fake-News-Detection-App.git
   cd Fake-News-Detection-App/fake-news-xai
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/Scripts/activate     # Windows Git Bash
   venv\Scripts\activate            # Windows Command Prompt
   source venv/bin/activate         # Mac / Linux
   ```

3. Install the Python packages:

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. Install ffmpeg (needed only for the audio and video inputs):

   ```bash
   winget install ffmpeg            # Windows
   brew install ffmpeg              # Mac
   sudo apt install ffmpeg          # Linux
   ```

   Close and reopen your terminal (and VS Code, if you use it), then check it with `ffmpeg -version`.

## Train the model

The trained model is not part of the source code, so train it before running the apps.

1. Download the **Fake and Real News Dataset** from Kaggle. It contains `Fake.csv` and `True.csv`. Put both files in `data/`.
2. Train:

   ```bash
   python train_model.py --fake data/Fake.csv --real data/True.csv
   ```

3. Optional, for better accuracy (takes noticeably longer):

   ```bash
   python tune_model.py --fake data/Fake.csv --real data/True.csv
   ```

Training saves `models/fake_news_pipeline.joblib` and `models/metrics.json` (accuracy, precision, recall, F1).

**Using a different dataset:**

```bash
python train_model.py --data data/mydata.csv --text_col text --label_col label
```

The label column must contain the words `FAKE` and `REAL`. TruthLens reads the model's `FAKE` probability by name, so numeric labels such as 0 and 1 must be converted first.

**Quick test without a real dataset:**

```bash
python data/make_sample_data.py
python train_model.py --data data/sample_news.csv --text_col text --label_col label
```

This tiny 20-row dataset only checks that the code runs. Its predictions are not meaningful.

## Run

**TruthLens:**

```bash
python server.py
```

Open http://localhost:5000. Keep the terminal open while you use the app.

**Streamlit:**

```bash
streamlit run app.py
```

If the `streamlit` command is not found, use `python -m streamlit run app.py`.

## API

The TruthLens pages call these endpoints on the Flask server.

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/metrics` | Model accuracy, F1 and dataset sizes |
| POST | `/api/analyze` | `{"text": "..."}` returns label, confidence, class probabilities and LIME word weights |
| POST | `/api/transcribe` | Upload an audio or video file, returns the transcript |
| POST | `/api/fetch-url` | `{"url": "..."}` returns the article title and text |
| POST | `/api/report` | Analysis result in, PDF report out |

## How it works

1. **Input:** pasted text, or text extracted from a link, or a transcript of audio or video.
2. **Cleaning:** lowercase, remove URLs, HTML and punctuation, drop stopwords.
3. **Classification:** TF-IDF features (single words and word pairs) feed a Logistic Regression classifier.
4. **Explanation:** LIME removes random words from the text, watches how the prediction changes, and fits a small linear model to estimate each word's influence.
5. **Output:** verdict, confidence, word weights and an optional PDF report.

Speech is transcribed with the free Google Web Speech API through the `SpeechRecognition` library, after ffmpeg converts the file. This needs an internet connection.

## Deployment

- **Streamlit version:** deploy on Streamlit Community Cloud. Commit `models/fake_news_pipeline.joblib`, `.streamlit/config.toml`, `requirements.txt` and `packages.txt`. Without the model file the app shows "No trained model found".
- **TruthLens (Flask) version:** Streamlit Cloud cannot host it. Use a Python host such as Render or PythonAnywhere, commit the model file, and run the app with a production server such as gunicorn instead of `python server.py`.

## Troubleshooting

| Problem | Fix |
|---|---|
| `streamlit: command not found` | Use `python -m streamlit run app.py` and make sure the virtual environment is active |
| scikit-learn fails to build (Cython or Meson error) | Run `python -m pip install --upgrade pip`, then install `scikit-learn>=1.8` |
| `Not Found` at http://localhost:5000 | The HTML, CSS and JS files must be inside the `frontend/` folder, with `setting.html` named `settings.html` |
| `ffmpeg` not found in the VS Code terminal | Fully close and reopen VS Code after installing ffmpeg |
| "No trained model found" | Run `train_model.py` first |
| Audio or video transcription fails | Check your internet connection, ffmpeg, and that the file contains clear speech |
| `git push` rejected (non-fast-forward) | Run `git pull origin main`, then push again |

## Current limitations

- The model learns writing style, not facts, so a well-written false story can pass. It is not a fact-checking tool.
- Accuracy depends on the training dataset and the period it covers.
- TruthLens login is a demo (no real accounts), and history is stored in the browser's localStorage.
- The TruthLens report page does not show word-level highlights yet. The `/api/analyze` endpoint already returns them, and the Streamlit app displays them.

## Ideas for future work

- Compare against a BERT model with SHAP explanations
- Cross-check claims against fact-checking databases
- Real user accounts and server-side history
- More languages and online deployment
