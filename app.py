import hashlib
import logging
import tempfile
import time
from pathlib import Path
from typing import Callable

import streamlit as st

from rag import chunk_text, extract_text_from_pdf, generate_answer, generate_embeddings, semantic_search


logger = logging.getLogger(__name__)


@st.cache_data(scope="session", max_entries=2, show_spinner=False)
def prepare_document(
	document_id: str,
	_pdf_bytes: bytes,
	_progress: Callable[[str], None],
) -> tuple[list[dict], object, int]:
	"""Extract, chunk, and embed a PDF while preserving its page metadata."""
	if len(document_id) != 64:
		raise ValueError("document_id must be a SHA-256 fingerprint.")

	temporary_path = None
	try:
		with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temporary_file:
			temporary_file.write(_pdf_bytes)
			temporary_path = temporary_file.name

		_progress("Extracting text from PDF...")

		def update_extraction_progress(current_page: int, total_pages: int) -> None:
			update_interval = max(1, total_pages // 20)
			if current_page == 1 or current_page == total_pages or current_page % update_interval == 0:
				_progress(f"Extracting text from PDF: page {current_page} of {total_pages}")

		logger.info("PDF extraction started.")
		stage_started = time.perf_counter()
		try:
			pages = extract_text_from_pdf(temporary_path, update_extraction_progress)
		except Exception as error:
			logger.error(
				"PDF extraction failed after %.2f seconds (%s).",
				time.perf_counter() - stage_started,
				type(error).__name__,
			)
			raise
		logger.info(
			"PDF extraction succeeded in %.2f seconds (%d pages).",
			time.perf_counter() - stage_started,
			len(pages),
		)

		_progress(f"Chunking text from {len(pages)} pages...")
		chunks = []
		update_interval = max(1, len(pages) // 20)
		logger.info("PDF chunking started.")
		stage_started = time.perf_counter()
		try:
			for page_index, page in enumerate(pages, start=1):
				page_chunks = chunk_text(page["text"])
				chunks.extend(
					{"text": text, "page_number": page["page_number"]}
					for text in page_chunks
				)
				if (
					page_index == 1
					or page_index == len(pages)
					or page_index % update_interval == 0
				):
					_progress(
						f"Chunking text: page {page_index} of {len(pages)} "
						f"({len(chunks)} chunks so far)"
					)
			if not chunks:
				raise ValueError("This PDF does not contain any text that can be searched.")
		except Exception as error:
			logger.error(
				"PDF chunking failed after %.2f seconds (%s).",
				time.perf_counter() - stage_started,
				type(error).__name__,
			)
			raise
		logger.info(
			"PDF chunking succeeded in %.2f seconds (%d chunks).",
			time.perf_counter() - stage_started,
			len(chunks),
		)

		_progress(
			f"Generating embeddings for {len(chunks)} text chunks. "
			"The first model download can take longer..."
		)
		logger.info("Document embedding generation started (%d chunks).", len(chunks))
		stage_started = time.perf_counter()
		try:
			embeddings = generate_embeddings(chunks)
		except Exception as error:
			logger.error(
				"Document embedding generation failed after %.2f seconds (%s).",
				time.perf_counter() - stage_started,
				type(error).__name__,
			)
			raise
		logger.info(
			"Document embedding generation succeeded in %.2f seconds.",
			time.perf_counter() - stage_started,
		)
		return chunks, embeddings, len(pages)
	finally:
		if temporary_path is not None:
			Path(temporary_path).unlink(missing_ok=True)


def select_suggested_question(question: str) -> None:
	"""Place a suggested question in the editable question field."""
	st.session_state["question_input"] = question


st.set_page_config(
	page_title="PDF RAG Chatbot",
	page_icon=":material/description:",
	layout="wide",
)

# These shell-level styles set the requested dark palette; components remain native Streamlit widgets.
st.html(
	"""
	<style>
		:root {
			color-scheme: dark;
			--primary-color: #8b7cf6;
			--background-color: #0b1120;
			--secondary-background-color: #111b2d;
			--text-color: #e5eaf4;
			--border-color: #263248;
		}
		.stApp,
		[data-testid="stAppViewContainer"] {
			background: #0b1120;
			color: #e5eaf4;
		}
		[data-testid="stHeader"] {
			background: rgba(11, 17, 32, 0.92);
		}
		section[data-testid="stSidebar"] {
			background: #101827;
			border-right: 1px solid #263248;
		}
		.main .block-container {
			max-width: 1180px;
			padding-top: 2rem;
			padding-bottom: 3rem;
		}
		.stButton button,
		.stFormSubmitButton button {
			border-radius: 0.75rem;
			min-height: 2.75rem;
			transition: border-color 120ms ease, background-color 120ms ease;
		}
		.stButton button:hover,
		.stFormSubmitButton button:hover {
			border-color: #7c6bf2;
		}
		[data-testid="stChatMessage"] {
			border: 1px solid #263248;
			border-radius: 1rem;
			padding: 1rem 1.1rem;
			background: #111b2d;
		}
		[data-testid="stTextInput"] input {
			border-radius: 0.75rem;
		}
		@media (max-width: 640px) {
			.main .block-container {
				padding-top: 1.25rem;
				padding-bottom: 2rem;
			}
			[data-testid="stChatMessage"] {
				padding: 0.8rem;
			}
		}
	</style>
	"""
)

st.session_state.setdefault("conversation_history", [])
st.session_state.setdefault("question_input", "")

with st.sidebar:
	st.markdown("### :material/folder_open: Document")
	st.caption("Upload a text-based PDF to ground your answers in its content.")
	uploaded_pdf = st.file_uploader(
		"Choose a PDF",
		type=["pdf"],
		help="PDF files only. Scanned image-only PDFs may not contain searchable text.",
	)
	document_status = st.empty()
	document_stats = st.empty()
	if uploaded_pdf is None:
		document_status.badge(
			"Awaiting a PDF",
			icon=":material/upload_file:",
			color="gray",
		)
		document_stats.caption("Page and chunk counts appear after processing.")
	else:
		document_status.badge(
			"Processing",
			icon=":material/pending:",
			color="blue",
		)
		st.caption(f"Selected: {uploaded_pdf.name}")

	st.caption("Answers are generated from the most relevant passages in your document.")

header_left, header_right = st.columns([5, 1], vertical_alignment="center")
with header_left:
	st.title("PDF RAG Chatbot", icon=":material/auto_awesome:")
	st.caption("Ask questions and get grounded answers from your PDF documents.")
header_status = header_right.empty()
if uploaded_pdf is None:
	header_status.badge(
		"No document loaded",
		icon=":material/radio_button_unchecked:",
		color="gray",
	)
else:
	header_status.badge(
		"Preparing document",
		icon=":material/pending:",
		color="blue",
	)

document_id = None
text_chunks = None
embeddings = None
page_count = None
if uploaded_pdf is not None:
	pdf_bytes = uploaded_pdf.getvalue()
	document_id = hashlib.sha256(pdf_bytes).hexdigest()

	try:
		with st.sidebar:
			with st.status("Preparing document...", expanded=True) as processing_status:
				def update_processing_status(message: str) -> None:
					processing_status.update(label=message, state="running", expanded=True)

				text_chunks, embeddings, page_count = prepare_document(
					document_id,
					pdf_bytes,
					update_processing_status,
				)
				processing_status.update(
					label=f"Document ready: {page_count} pages, {len(text_chunks)} text chunks",
					state="complete",
					expanded=False,
				)
	except ValueError as error:
		message = str(error)
		if "no extractable text" in message.lower():
			st.error("No text could be extracted. Try a text-based PDF rather than a scanned image.")
		else:
			st.error("This PDF could not be read. Check that it is valid and not password-protected.")
		document_status.badge("Processing failed", icon=":material/error:", color="red")
		document_id = None
	except Exception:
		st.error("The PDF could not be processed. Please try another file.")
		document_status.badge("Processing failed", icon=":material/error:", color="red")
		document_id = None

	if document_id is not None:
		document_status.badge("Ready", icon=":material/check_circle:", color="green")
		with document_stats.container(border=True):
			st.markdown(f"**{uploaded_pdf.name}**")
			metric_pages, metric_chunks = st.columns(2)
			metric_pages.metric("Pages", page_count)
			metric_chunks.metric("Text chunks", len(text_chunks))
		header_status.badge(
			"Document ready",
			icon=":material/check_circle:",
			color="green",
		)
	else:
		header_status.badge(
			"Document needs attention",
			icon=":material/error:",
			color="red",
		)

st.space("medium")
st.subheader("Conversation", icon=":material/chat:")

if (
	document_id is not None
	and st.session_state.get("answer_document_id") == document_id
	and isinstance(st.session_state.get("last_question"), str)
	and isinstance(st.session_state.get("last_answer"), str)
	and not any(
		message["document_id"] == document_id
		and message["question"] == st.session_state["last_question"]
		for message in st.session_state["conversation_history"]
	)
):
	st.session_state["conversation_history"].append(
		{
			"document_id": document_id,
			"question": st.session_state["last_question"],
			"answer": st.session_state["last_answer"],
		}
	)

document_history = [
	message
	for message in st.session_state["conversation_history"]
	if message["document_id"] == document_id
]

if document_id is None:
	with st.container(border=True):
		st.markdown("#### Your document workspace")
		st.write("Upload a PDF in the sidebar to prepare it for question answering.")
		st.caption("Text extraction, chunking, and embedding progress will appear in the document panel.")
elif not document_history:
	with st.container(border=True):
		st.markdown("#### Start with a question")
		st.write("Ask about key ideas, important details, or conclusions in your document.")
		st.caption("Try one of these examples, then review or edit it before submitting.")

	suggestions = [
		"What is this document about?",
		"What are the most important points?",
		"What conclusions does the document reach?",
	]
	suggestion_columns = st.columns(3)
	for index, suggestion in enumerate(suggestions):
		suggestion_columns[index].button(
			suggestion,
			key=f"suggested_question_{index}",
			icon=":material/arrow_outward:",
			width="stretch",
			help="Add this question to the input below. You can edit it before asking.",
			on_click=select_suggested_question,
			args=(suggestion,),
		)

for message in document_history:
	with st.chat_message("user", avatar=":material/person:"):
		st.markdown(message["question"])
	with st.chat_message("assistant", avatar=":material/auto_awesome:"):
		st.markdown(message["answer"])

if document_id is not None and text_chunks is not None and embeddings is not None:
	with st.form("question_form", clear_on_submit=True):
		question = st.text_input(
			"Your question",
			key="question_input",
			placeholder="Ask something about this PDF...",
		)
		submitted = st.form_submit_button(
			"Ask",
			type="primary",
			icon=":material/arrow_upward:",
			width="stretch",
		)

	if submitted:
		if not question.strip():
			st.error("Enter a question before selecting Ask.")
		else:
			try:
				with st.chat_message("user", avatar=":material/person:"):
					st.markdown(question)
				with st.chat_message("assistant", avatar=":material/auto_awesome:"):
					with st.status(
						"Finding relevant passages...",
						type="step",
						expanded=True,
					) as retrieval_status:
						logger.info("Question retrieval and query embedding started.")
						stage_started = time.perf_counter()
						try:
							retrieved_chunks = semantic_search(question, text_chunks, embeddings)
						except Exception as error:
							logger.error(
								"Question retrieval and query embedding failed after %.2f seconds (%s).",
								time.perf_counter() - stage_started,
								type(error).__name__,
							)
							raise
						logger.info(
							"Question retrieval and query embedding succeeded in %.2f seconds (%d results).",
							time.perf_counter() - stage_started,
							len(retrieved_chunks),
						)
						retrieval_status.update(
							label="Relevant passages found",
							state="complete",
							expanded=False,
						)

					if not retrieved_chunks:
						st.error("No relevant text was found. Try rephrasing your question.")
					else:
						with st.status(
							"Generating an answer...",
							type="step",
							expanded=True,
						) as answer_status:
							answer = generate_answer(question, retrieved_chunks)
							answer_status.update(
								label="Answer ready",
								state="complete",
								expanded=False,
							)
						st.markdown(answer)
						st.session_state["last_answer"] = answer
						st.session_state["last_question"] = question
						st.session_state["answer_document_id"] = document_id
						st.session_state["conversation_history"].append(
							{
								"document_id": document_id,
								"question": question,
								"answer": answer,
							}
						)
			except ValueError as error:
				if "GEMINI_API_KEY is not set" in str(error):
					st.error("Gemini is not configured. Add GEMINI_API_KEY to your local .env file and restart the app.")
				else:
					st.error("The question could not be processed. Please check it and try again.")
			except RuntimeError:
				st.error("Gemini could not generate an answer. Check your API configuration or try again shortly.")
			except Exception:
				st.error("Something went wrong while searching the PDF. Please try again.")
