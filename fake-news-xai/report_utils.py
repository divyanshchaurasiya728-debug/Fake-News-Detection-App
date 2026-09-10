"""
report_utils.py
Generates a downloadable PDF report summarizing one analysis result:
the input text, prediction, confidence, and top contributing words.
"""

from datetime import datetime

from fpdf import FPDF


class ReportPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(108, 92, 231)  # matches app's purple accent
        self.cell(0, 10, "Explainable Fake News Detector - Report", ln=True, align="C")
        self.set_text_color(0, 0, 0)
        self.set_font("Helvetica", "", 9)
        self.cell(0, 6, datetime.now().strftime("Generated: %Y-%m-%d %H:%M"), ln=True, align="C")
        self.ln(4)


def _safe(text: str) -> str:
    """FPDF's core fonts only support latin-1; strip anything outside that range."""
    return text.encode("latin-1", "ignore").decode("latin-1")


def generate_pdf_report(
    article_text: str,
    label: str,
    confidence: float,
    class_probs: dict,
    word_weights: list,
    source: str = "Pasted text",
) -> bytes:
    """
    word_weights: list of (word, weight) tuples, as returned by
                  LimeTextExplainer's .as_list()
    Returns raw PDF bytes, ready to hand to Streamlit's download_button.
    """
    pdf = ReportPDF()
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, f"Source: {_safe(source)}", ln=True)
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 13)
    color = (220, 50, 50) if label.upper() == "FAKE" else (40, 160, 80)
    pdf.set_text_color(*color)
    pdf.cell(0, 10, f"Prediction: {label}  ({confidence*100:.1f}% confidence)", ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Class probabilities:", ln=True)
    pdf.set_font("Helvetica", "", 10)
    for cls, p in class_probs.items():
        pdf.cell(0, 6, f"  - {cls}: {p*100:.1f}%", ln=True)
    pdf.ln(3)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Top contributing words (LIME):", ln=True)
    pdf.set_font("Helvetica", "", 10)
    for word, weight in word_weights:
        direction = "toward FAKE" if weight < 0 else "toward REAL"
        pdf.cell(0, 6, f"  - {_safe(word)}: {weight:+.3f} ({direction})", ln=True)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Analyzed text:", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 6, _safe(article_text))
    pdf.ln(6)

    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(120, 120, 120)
    pdf.multi_cell(
        0, 5,
        "Disclaimer: This is a student/demo project, not a fact-checking authority. "
        "Predictions reflect writing-style patterns learned from training data, not verified facts."
    )

    return bytes(pdf.output())
