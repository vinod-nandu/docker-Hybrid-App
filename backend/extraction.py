"""PDF text extraction using PyMuPDF (fitz)."""
import fitz  # PyMuPDF


def extract_pdf_pages(file_path: str) -> list[dict]:
    """Return a list of {page: int, text: str} for every non-empty page."""
    pages = []
    with fitz.open(file_path) as doc:
        for i, page in enumerate(doc):
            text = page.get_text("text").strip()
            if text:
                pages.append({"page": i + 1, "text": text})
    return pages
