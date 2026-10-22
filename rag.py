import re

import fitz


def extract_text_from_pdf(pdf_path: str) -> list[dict]:
	"""Extract normalized text from each page of a PDF.

	Each result includes a 1-based page number and that page's text with
	runs of whitespace collapsed to single spaces.
	"""
	try:
		with fitz.open(pdf_path) as pdf:
			pages = [
				{
					"page_number": page_number,
					"text": re.sub(r"\s+", " ", page.get_text("text")).strip(),
				}
				for page_number, page in enumerate(pdf, start=1)
			]
	except (OSError, RuntimeError, ValueError) as exc:
		raise ValueError(f"Unable to read PDF {pdf_path!r}: {exc}") from exc

	if not any(page["text"] for page in pages):
		raise ValueError(f"PDF contains no extractable text: {pdf_path!r}")

	return pages


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
	"""Split text into character-based chunks with overlap between neighbors.

	Returns an empty list for empty or whitespace-only text. Overlap helps
	retain context at chunk boundaries while keeping chunks within the size.
	"""
	if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size <= 0:
		raise ValueError("chunk_size must be a positive integer.")
	if not isinstance(overlap, int) or isinstance(overlap, bool) or overlap < 0:
		raise ValueError("overlap must be a non-negative integer.")
	if overlap >= chunk_size:
		raise ValueError("overlap must be smaller than chunk_size.")
	if not text or not text.strip():
		return []

	step = chunk_size - overlap
	chunks = []
	start = 0
	while start < len(text):
		chunks.append(text[start : start + chunk_size])
		if start + chunk_size >= len(text):
			break
		start += step

	return chunks
