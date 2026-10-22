from functools import lru_cache
import os
import re
from typing import Any

import fitz
import numpy as np
from dotenv import load_dotenv
from google import genai


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


@lru_cache(maxsize=1)
def _get_embedding_model():
	"""Load and reuse the sentence-transformers model on first use."""
	from sentence_transformers import SentenceTransformer

	return SentenceTransformer(EMBEDDING_MODEL_NAME)


def _get_chunk_text(chunk: str | dict[str, Any]) -> str:
	"""Return the text field from a plain string or a chunk record."""
	if isinstance(chunk, str):
		return chunk
	if not isinstance(chunk, dict):
		raise TypeError("Each chunk must be a string or a dictionary with a 'text' field.")
	if not isinstance(chunk.get("text"), str):
		raise ValueError("Each chunk dictionary must contain a string 'text' field.")
	return chunk["text"]


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


def generate_embeddings(text_chunks: list[str | dict[str, Any]]) -> np.ndarray:
	"""Convert text chunks into vectors using a sentence-transformers model.

	A vector is a numerical representation of a chunk's meaning. Chunk records
	may include metadata such as a page number; only their ``text`` is encoded.
	"""
	if not isinstance(text_chunks, list):
		raise TypeError("text_chunks must be a list.")

	chunk_texts = [_get_chunk_text(chunk) for chunk in text_chunks]
	if not chunk_texts:
		return np.empty((0, 0), dtype=np.float32)

	encoded = _get_embedding_model().encode(
		chunk_texts,
		convert_to_numpy=True,
		show_progress_bar=False,
	)
	try:
		embeddings = np.asarray(encoded, dtype=np.float32)
	except (TypeError, ValueError) as exc:
		raise ValueError("The embedding model returned invalid numerical data.") from exc

	if embeddings.ndim == 1 and len(chunk_texts) == 1:
		embeddings = embeddings.reshape(1, -1)
	if embeddings.ndim != 2 or embeddings.shape[0] != len(chunk_texts):
		raise ValueError("The embedding model must return one vector per text chunk.")
	if embeddings.shape[1] == 0 or not np.isfinite(embeddings).all():
		raise ValueError("Embeddings must contain finite numerical values.")

	return embeddings


def semantic_search(
	query: str,
	text_chunks: list[str | dict[str, Any]],
	embeddings: np.ndarray | list[list[float]],
	top_k: int = 3,
) -> list[dict[str, Any]]:
	"""Return the chunks most similar to a query, with cosine scores.

	Cosine similarity compares the angle between the query vector and each
	chunk vector. Results include the original chunk text and any metadata; a
	``similarity`` value is added to each result. Zero vectors receive a score
	of zero instead of causing a division error.
	"""
	if not isinstance(query, str):
		raise TypeError("query must be a string.")
	if not isinstance(text_chunks, list):
		raise TypeError("text_chunks must be a list.")
	if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
		raise ValueError("top_k must be a positive integer.")

	chunk_texts = [_get_chunk_text(chunk) for chunk in text_chunks]
	if not query.strip() or not chunk_texts:
		return []

	try:
		chunk_vectors = np.asarray(embeddings, dtype=np.float32)
	except (TypeError, ValueError) as exc:
		raise ValueError("embeddings must be a 2D array of numbers.") from exc
	if chunk_vectors.ndim != 2 or chunk_vectors.shape[0] != len(chunk_texts):
		raise ValueError("embeddings must contain one vector per text chunk.")
	if chunk_vectors.shape[1] == 0 or not np.isfinite(chunk_vectors).all():
		raise ValueError("embeddings must contain finite numerical vectors.")

	encoded_query = _get_embedding_model().encode(
		[query],
		convert_to_numpy=True,
		show_progress_bar=False,
	)
	try:
		query_vector = np.asarray(encoded_query, dtype=np.float32)
	except (TypeError, ValueError) as exc:
		raise ValueError("The embedding model returned an invalid query vector.") from exc
	if query_vector.ndim == 2 and query_vector.shape[0] == 1:
		query_vector = query_vector[0]
	if query_vector.ndim != 1 or query_vector.shape[0] != chunk_vectors.shape[1]:
		raise ValueError("The query embedding dimension must match the chunk embeddings.")
	if not np.isfinite(query_vector).all():
		raise ValueError("The query embedding must contain finite numerical values.")

	chunk_norms = np.linalg.norm(chunk_vectors, axis=1)
	query_norm = np.linalg.norm(query_vector)
	denominators = chunk_norms * query_norm
	similarities = np.divide(
		chunk_vectors @ query_vector,
		denominators,
		out=np.zeros_like(chunk_norms),
		where=denominators != 0,
	)
	ranked_indexes = np.argsort(-similarities, kind="stable")[:top_k]

	results = []
	for index in ranked_indexes:
		chunk = text_chunks[int(index)]
		result = {"text": chunk} if isinstance(chunk, str) else dict(chunk)
		result["similarity"] = float(similarities[index])
		results.append(result)

	return results


def generate_answer(
	question: str,
	retrieved_chunks: list[str | dict[str, Any]],
) -> str:
	"""Ask Gemini to answer a question using retrieved PDF context.

	The supplied chunks ground the response in relevant PDF text. Source page
	labels are included when available; plain strings are used without inferred
	page numbers. The API key is read from the environment or a local .env file.
	"""
	if not isinstance(question, str):
		raise TypeError("question must be a string.")
	if not question.strip():
		raise ValueError("question cannot be empty.")
	if not isinstance(retrieved_chunks, list):
		raise TypeError("retrieved_chunks must be a list.")

	context_parts = []
	for chunk in retrieved_chunks:
		chunk_text = _get_chunk_text(chunk)
		if not chunk_text.strip():
			continue
		if isinstance(chunk, dict) and chunk.get("page_number") is not None:
			context_parts.append(f"[Source page: {chunk['page_number']}]\n{chunk_text}")
		else:
			context_parts.append(chunk_text)

	if not context_parts:
		raise ValueError("retrieved_chunks must contain non-empty text.")

	load_dotenv(override=False)
	api_key = os.getenv("GEMINI_API_KEY")
	if not api_key or not api_key.strip():
		raise ValueError("GEMINI_API_KEY is not set. Add it to your environment or .env file.")

	model_name = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip()
	if not model_name:
		model_name = DEFAULT_GEMINI_MODEL

	prompt = (
		"Answer the user's question using only the supplied PDF context. "
		"Do not invent facts or add unsupported details. If the context does not "
		"contain enough information, clearly say so.\n\n"
		f"PDF context:\n{chr(10).join(context_parts)}\n\n"
		f"User question:\n{question.strip()}\n\n"
		"Answer:"
	)

	try:
		client = genai.Client(api_key=api_key)
		response = client.models.generate_content(model=model_name, contents=prompt)
	except Exception:
		raise RuntimeError(
			"Gemini API request failed. Check the API configuration and try again."
		) from None

	answer = response.text
	if not isinstance(answer, str) or not answer.strip():
		raise RuntimeError("Gemini returned an empty answer.")

	return answer.strip()
