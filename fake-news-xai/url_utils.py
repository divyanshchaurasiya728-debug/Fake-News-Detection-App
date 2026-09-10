"""
url_utils.py
Fetches a news article's main text content from a URL, so the app can
accept a link instead of pasted text.

Uses a simple, dependency-light approach (requests + BeautifulSoup)
rather than a heavier library, so it stays easy to deploy.
"""

import re

import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}


def fetch_article_from_url(url: str, timeout: int = 10) -> dict:
    """
    Downloads the page at `url` and extracts a best-effort title + body text.

    Returns: {"title": str, "text": str}
    Raises: requests.RequestException on network/HTTP errors,
            ValueError if no meaningful article text could be extracted.
    """
    response = requests.get(url, headers=HEADERS, timeout=timeout)
    response.raise_for_status()

    soup = BeautifulSoup(response.content, "html.parser")

    # Remove elements unlikely to contain article body text
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form", "iframe"]):
        tag.decompose()

    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    else:
        h1 = soup.find("h1")
        if h1:
            title = h1.get_text(strip=True)

    # Prefer an <article> tag if present; otherwise fall back to all <p> tags
    article_tag = soup.find("article")
    paragraphs = article_tag.find_all("p") if article_tag else soup.find_all("p")

    text_blocks = []
    for p in paragraphs:
        text = p.get_text(" ", strip=True)
        # Skip short/boilerplate lines (nav links, captions, cookie notices, etc.)
        if len(text.split()) >= 6:
            text_blocks.append(text)

    body_text = "\n\n".join(text_blocks)
    body_text = re.sub(r"\n{3,}", "\n\n", body_text).strip()

    if len(body_text.split()) < 20:
        raise ValueError(
            "Could not extract enough article text from this URL. "
            "The site may block automated requests, or use a layout this "
            "extractor doesn't recognize. Try pasting the article text directly instead."
        )

    return {"title": title, "text": body_text}
