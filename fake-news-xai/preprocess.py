"""
preprocess.py
Shared text-cleaning utilities used by both train_model.py and app.py.
Keeping this logic in one place ensures the app cleans text
EXACTLY the same way the model was trained on.
"""

import re
import string

# A small, dependency-free stopword list so this file works
# even before nltk's stopword corpus is downloaded.
BASIC_STOPWORDS = set("""
a an the and or but if while is are was were be been being
to of in on for with as by at from this that these those it its
he she they them his her their you your i we our us not no nor
do does did doing have has had having will would shall should
can could may might must than then so such only own same too very
""".split())


def clean_text(text: str) -> str:
    """
    Basic, transparent cleaning pipeline:
    1. Lowercase
    2. Remove URLs
    3. Remove HTML tags
    4. Remove punctuation/numbers
    5. Remove extra whitespace
    6. Remove basic stopwords

    Kept intentionally simple (no stemming) so that LIME explanations
    later show real, readable words instead of stemmed fragments.
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)          # URLs
    text = re.sub(r"<.*?>", " ", text)                       # HTML tags
    text = re.sub(r"[^a-z\s]", " ", text)                    # punctuation/numbers
    text = re.sub(r"\s+", " ", text).strip()                 # extra whitespace

    words = [w for w in text.split() if w not in BASIC_STOPWORDS and len(w) > 2]
    return " ".join(words)
