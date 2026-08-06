# Explainable AI for Fake News Detection

A fake news classifier (TF-IDF + Logistic Regression) with a Streamlit app
that explains *why* each prediction was made, using LIME.

## Project structure

```
fake-news-xai/
├── requirements.txt
├── preprocess.py          # shared text-cleaning logic
├── train_model.py         # trains + saves the model
├── app.py                 # Streamlit demo app
├── data/
│   └── make_sample_data.py  # generates a tiny synthetic dataset for testing
└── models/                # trained model + metrics get saved here
```

## 1. Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Get a dataset

**Option A — quick test (synthetic, 20 rows, just to confirm the code runs):**
```bash
python data/make_sample_data.py
python train_model.py --data data/sample_news.csv --text_col text --label_col label
```

**Option B — real dataset (recommended for an actual project/report):**
Download the Kaggle "Fake and Real News Dataset"
(https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset),
which gives you `Fake.csv` and `True.csv`. Put them in `data/`, then:
```bash
python train_model.py --fake data/Fake.csv --real data/True.csv
```

**Option C — any other dataset** (LIAR, FakeNewsNet, your own CSV) as long as
it has a text column and a label column:
```bash
python train_model.py --data data/your_data.csv --text_col <col> --label_col <col>
```

This prints Accuracy/Precision/Recall/F1 and saves:
- `models/fake_news_pipeline.joblib`
- `models/metrics.json`

## 3. Run the app

```bash
streamlit run app.py
```

Opens in your browser (usually http://localhost:8501). Paste in a headline
or article, hit **Analyze**, and you'll see:
- The predicted label (FAKE/REAL) with a confidence score
- The original text with words highlighted green (pushes toward REAL)
  or red (pushes toward FAKE)
- A bar chart of the top contributing words

## 4. Ideas to extend this into a stronger report/project

- Swap Logistic Regression for a BERT model, and use `transformers-interpret`
  or attention visualization (`bertviz`) instead of LIME, and compare
  explanation quality between the two.
- Add SHAP alongside LIME and compare which words each method flags.
- Add a "faithfulness" evaluation: remove the top-K words LIME flagged and
  check how much the prediction confidence actually drops.
- Log user feedback ("was this explanation helpful?") for a small human
  evaluation section in your report.

## Notes

- This is a demo/student project, not a fact-checking tool. It detects
  *writing-style patterns* correlated with fake news in the training data —
  it does not verify facts.
- Accuracy depends heavily on your dataset; the included sample dataset
  is only 20 rows and exists purely to confirm the code runs end-to-end.
