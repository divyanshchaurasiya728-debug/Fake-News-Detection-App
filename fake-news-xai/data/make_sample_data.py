"""
make_sample_data.py
Generates a tiny synthetic dataset so you can run the whole pipeline
end-to-end BEFORE plugging in a real dataset (e.g. Kaggle's
"Fake and Real News Dataset" -> Fake.csv + True.csv, or the LIAR dataset).

This is only for testing that the code runs. Replace with real data
for any actual results.

Usage:
    python data/make_sample_data.py
"""

import pandas as pd
import os

real_samples = [
    "The central bank raised interest rates by a quarter point on Wednesday, citing persistent inflation pressures across the economy.",
    "Local officials announced a new public transit line connecting the eastern suburbs to downtown, with construction set to begin next spring.",
    "Researchers published a peer-reviewed study in a major journal showing a modest decline in regional air pollution over the past decade.",
    "The city council approved the annual budget after months of public hearings and revisions to the proposed spending plan.",
    "A new report from the national statistics agency shows unemployment held steady last month across most sectors of the economy.",
    "The hospital opened a new wing dedicated to pediatric care, funded partly by a state grant and private donations.",
    "Election officials confirmed that ballots will be counted starting Tuesday evening, with results expected within 48 hours.",
    "The university announced a partnership with a local college to expand access to affordable STEM education programs.",
    "Weather services issued a routine advisory for heavy rainfall expected across the coastal region this weekend.",
    "The company reported quarterly earnings in line with analyst expectations, with revenue growth driven by its cloud division.",
]

fake_samples = [
    "Secret government memo PROVES they are hiding the truth about the moon landing, insiders reveal shocking details!",
    "Doctors HATE this one weird trick that cures all diseases instantly, Big Pharma doesn't want you to know!",
    "BREAKING: Celebrity secretly replaced by clone, anonymous source says this changes everything you thought you knew!",
    "Scientists confirm the earth will stop spinning next month, government refuses to warn the public about the disaster!",
    "Shocking leaked documents show world leaders secretly control the weather using hidden machines, share before it's deleted!",
    "This miracle fruit melts fat overnight, doctors are furious that this natural cure is being suppressed by corporations!",
    "Anonymous whistleblower claims aliens have been living among us for decades, mainstream media refuses to cover this story!",
    "You won't believe what they found buried under the White House, officials are desperately trying to cover it up!",
    "New study 'proves' vaccines contain microchips that track your every move, share this before it gets banned!",
    "Local man discovers ancient prophecy that predicts the exact date the world will end, experts are in stunned silence!",
]

def main():
    rows = []
    for t in real_samples:
        rows.append({"text": t, "label": "REAL"})
    for t in fake_samples:
        rows.append({"text": t, "label": "FAKE"})

    df = pd.DataFrame(rows)
    out_path = os.path.join(os.path.dirname(__file__), "sample_news.csv")
    df.to_csv(out_path, index=False)
    print(f"Sample dataset written to {out_path} ({len(df)} rows)")
    print("Replace this with a real dataset (Kaggle Fake/Real News, LIAR, FakeNewsNet, etc.) for real results.")

if __name__ == "__main__":
    main()
