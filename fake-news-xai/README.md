# Explainable AI for Fake News Detection

A fake news classifier (TF-IDF + Logistic Regression) with a Streamlit app
that predicts FAKE vs REAL from **text, a URL, an audio clip, or a video file**,
and explains *why* each prediction was made using LIME. Results can be
downloaded as a PDF report.

## Project structure

```
fake-news-xai/
├── requirements.txt
├── packages.txt            # system packages for deployment (ffmpeg)
├── .streamlit/
│   └── config.toml         # app theme
├── preprocess.py           # shared text-cleaning logic
├── train_model.py          # trains + saves the model
├── tune_model.py           # hyperparameter-tuned training (optional, slower)
├── url_utils.py            # extracts article text from a URL
├── media_utils.py          # transcribes audio/video to text
├── report_utils.py         # generates the downloadable PDF report
├── app.py                  # Streamlit app (LIME-based, main app)
├── data/
│   └── make_sample_data.py   # generates a tiny synthetic dataset for testing
└── models/                 # trained model + metrics get saved here
```

## 1. Setup

**Install ffmpeg first** (required for the Audio/Video input tabs):
- Windows: `winget install ffmpeg` (or see manual install instructions if that fails)
- Mac: `brew install ffmpeg`
- Linux: `sudo apt install ffmpeg`

Verify it worked:
```bash
ffmpeg -version
```

Then set up the Python environment:
```bash
python -m venv venv
source venv/Scripts/activate      # Windows Git Bash; use venv\Scripts\activate for cmd/PowerShell
pip install -r requirements.txt
```

## 2. Get a dataset and train the model

**Quick test (synthetic, 20 rows, just to confirm the code runs):**
```bash
python data/make_sample_data.py
python train_model.py --data data/sample_news.csv --text_col text --label_col label
```

**Real dataset (recommended):**
Download the Kaggle "Fake and Real News Dataset"
(search "Fake and Real News Dataset" by Clément Bisaillon on Kaggle),
which gives you `Fake.csv` and `True.csv`. Put both in `data/`, then:
```bash
python train_model.py --fake data/Fake.csv --real data/True.csv
```

**For better accuracy**, use the hyperparameter-tuned version instead (slower, searches multiple settings):
```bash
python tune_model.py --fake data/Fake.csv --real data/True.csv
```

## 3. Run the app

```bash
streamlit run app.py
```
(or `python -m streamlit run app.py` if the `streamlit` command isn't found on your PATH)

Opens at `http://localhost:8501`. The app has four input tabs:

### 📝 Text
Paste a headline or article directly, or click one of the example chips to auto-fill the box.

### 🔗 URL
Paste a link to a news article. The app fetches the page and extracts the headline + body text automatically. Works best on standard article layouts — heavily JavaScript-rendered or paywalled sites may not extract cleanly.

### 🎵 Audio
Upload an audio clip (`.wav`, `.mp3`, `.m4a`, `.ogg`, `.flac`). The app transcribes it to text using free speech recognition, then runs the transcript through the classifier. Requires an internet connection (the transcription service is cloud-based) and a working ffmpeg install.

### 🎬 Video
Upload a video file (`.mp4`, `.mov`, `.mkv`, `.avi`, `.webm`). The app extracts the audio track, transcribes it, and classifies the transcript — same underlying process as the Audio tab.

After any input method, hit **Analyze** to see:
- The predicted label (FAKE/REAL) with a confidence score
- The text with words highlighted green (pushes toward REAL) or red (pushes toward FAKE)
- A bar chart of the top contributing words
- A **Download PDF report** button with the full result summary

## 4. Deploying to Streamlit Community Cloud

1. Push your repo to GitHub, making sure these are committed (not excluded by `.gitignore`):
   - `models/fake_news_pipeline.joblib` (the trained model)
   - `.streamlit/config.toml` (theme)
   - `packages.txt` (installs ffmpeg on the server — required for Audio/Video tabs to work when deployed)
2. Go to **share.streamlit.io**, sign in with GitHub, click **"Create app"**.
3. Select your repo, branch `main`, main file path `app.py`, then **Deploy**.

## 5. Ideas to extend this further

- Swap Logistic Regression for a BERT model, and use `transformers-interpret`
  or attention visualization (`bertviz`) instead of LIME, and compare
  explanation quality between the two.
- Add SHAP alongside LIME and compare which words each method flags.
- Add a "faithfulness" evaluation: remove the top-K words LIME flagged and
  check how much the prediction confidence actually drops.
- Batch/CSV upload mode to analyze many articles at once.

## Notes

- This is a demo/student project, not a fact-checking tool. It detects
  *writing-style patterns* correlated with fake news in the training data —
  it does not verify facts.
- Accuracy depends heavily on your dataset; the included sample dataset
  is only 20 rows and exists purely to confirm the code runs end-to-end.
- Audio/video transcription accuracy depends on recording clarity and
  requires internet access to the speech recognition service.
