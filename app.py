import hashlib
import logging
import tempfile
import time
from pathlib import Path
from typing import Callable

import streamlit as st

from conversation_store import (
	clear_history,
	create_conversation,
	get_conversation,
	get_messages,
	initialize_database,
	list_conversations,
	save_message,
)
from rag import (
	GeminiUnavailableError,
	chunk_text,
	extract_text_from_pdf,
	generate_answer,
	generate_embeddings,
	semantic_search,
)


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
	page_title="PDF RAG Assistant",
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
			background:
				radial-gradient(ellipse at 15% 0%, rgba(82, 73, 190, 0.18), transparent 38%),
				radial-gradient(ellipse at 92% 22%, rgba(24, 111, 168, 0.12), transparent 32%),
				#0b1120;
			color: #e5eaf4;
		}
		[data-testid="stHeader"] {
			background: rgba(11, 17, 32, 0.72);
			backdrop-filter: blur(14px);
		}
		section[data-testid="stSidebar"] {
			background: rgba(16, 24, 39, 0.88);
			border-right: 1px solid rgba(148, 163, 184, 0.16);
			backdrop-filter: blur(18px);
		}
		.main .block-container {
			max-width: 1320px;
			padding-top: 2rem;
			padding-bottom: 3rem;
		}
		[data-testid="stFileUploaderDropzone"] {
			background: rgba(28, 39, 61, 0.48);
			border: 1px dashed rgba(139, 124, 246, 0.58);
			border-radius: 1rem;
			transition: border-color 160ms ease, background-color 160ms ease;
		}
		[data-testid="stFileUploaderDropzone"]:hover {
			background: rgba(57, 55, 106, 0.34);
			border-color: #a89cff;
		}
		.stButton button,
		.stFormSubmitButton button {
			border-radius: 0.75rem;
			min-height: 2.75rem;
			transition: border-color 160ms ease, background-color 160ms ease, transform 160ms ease;
		}
		.stButton button:hover,
		.stFormSubmitButton button:hover {
			border-color: #a89cff;
			transform: translateY(-1px);
		}
		[data-testid="stChatMessage"] {
			border: 1px solid rgba(148, 163, 184, 0.18);
			border-radius: 1rem;
			padding: 1rem 1.1rem;
			background: linear-gradient(145deg, rgba(27, 39, 61, 0.84), rgba(17, 27, 45, 0.72));
			backdrop-filter: blur(14px);
			animation: message-enter 220ms ease-out both;
		}
		[data-testid="stStatusWidget"],
		[data-testid="stVerticalBlockBorderWrapper"] {
			border-color: rgba(148, 163, 184, 0.18);
			border-radius: 1rem;
		}
		[data-testid="stTextInput"] input {
			border-radius: 0.75rem;
		}
		.app-brand {
			display: flex;
			align-items: center;
			gap: 0.9rem;
			min-height: 3.5rem;
			margin: 0 0 0.2rem;
		}
		.app-brand__icon {
			display: grid;
			width: 3rem;
			height: 3rem;
			flex: 0 0 3rem;
			place-items: center;
			border: 1px solid rgba(167, 139, 250, 0.42);
			border-radius: 0.95rem;
			background: linear-gradient(140deg, rgba(124, 107, 242, 0.24), rgba(46, 157, 205, 0.16));
			box-shadow: 0 8px 28px rgba(76, 65, 180, 0.16);
			font-size: 1.65rem;
			line-height: 1;
		}
		.app-brand__title {
			margin: 0;
			background: linear-gradient(100deg, #f4f1ff 0%, #c7c1ff 55%, #9bdcf3 100%);
			background-clip: text;
			color: transparent;
			font-size: clamp(1.55rem, 3vw, 2.15rem);
			font-weight: 700;
			letter-spacing: -0.035em;
			line-height: 1.15;
		}
		@keyframes message-enter {
			from { opacity: 0; transform: translateY(7px); }
			to { opacity: 1; transform: translateY(0); }
		}
		@media (prefers-reduced-motion: reduce) {
			*, *::before, *::after {
				animation-duration: 0.01ms !important;
				animation-iteration-count: 1 !important;
				scroll-behavior: auto !important;
				transition-duration: 0.01ms !important;
			}
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

try:
	initialize_database()
except Exception as error:
	logger.error("Could not initialize local conversation history (%s).", type(error).__name__)
	st.error("Local conversation history is unavailable. Check the application's data directory.")
	st.stop()

st.session_state.setdefault("last_uploaded_file_id", None)
st.session_state.setdefault("selected_conversation_id", None)
st.session_state.setdefault("question_input", "")
st.session_state.setdefault("upload_widget_generation", 0)


@st.dialog("Clear saved history?")
def confirm_clear_history() -> None:
	st.warning(
		"This permanently deletes all saved conversations, questions, answers, "
		"and document records from this app."
	)
	if st.button(
		"Delete all saved history",
		type="primary",
		icon=":material/delete_forever:",
		width="stretch",
	):
		try:
			clear_history()
		except Exception as error:
			logger.error(
				"Could not clear saved conversations (%s).",
				type(error).__name__,
			)
			st.error("Saved history could not be cleared. Please try again.")
			return

		st.session_state["selected_conversation_id"] = None
		st.session_state["last_uploaded_file_id"] = None
		st.session_state["question_input"] = ""
		st.session_state["upload_widget_generation"] += 1
		st.rerun()

	if st.button("Cancel", width="stretch"):
		st.rerun()

with st.sidebar:
	st.markdown("### :material/folder_open: Your documents")
	st.caption("Upload a text-based PDF to ground your answers in its content.")
	uploaded_pdf = st.file_uploader(
		"Choose a PDF",
		type=["pdf"],
		help="PDF files only. Scanned image-only PDFs may not contain searchable text.",
		key=f"pdf_uploader_{st.session_state['upload_widget_generation']}",
	)
	document_status = st.empty()
	document_stats = st.empty()
	if uploaded_pdf is None:
		document_status.badge(
			"Upload a PDF to start",
			icon=":material/upload_file:",
			color="gray",
		)
	document_stats.caption("Your past documents and conversations are saved locally.")

header_left, header_right = st.columns([5, 1], vertical_alignment="center")
with header_left:
	st.html(
		"""
		<header class="app-brand" aria-label="PDF RAG Assistant">
			<span class="app-brand__icon" aria-hidden="true">🤖</span>
			<h1 class="app-brand__title">PDF RAG Assistant</h1>
		</header>
		"""
	)
	st.caption("Ask questions and get grounded answers from your PDF documents.")
header_status = header_right.empty()
header_status.badge(
	"Choose a conversation",
	icon=":material/radio_button_unchecked:",
	color="gray",
)

uploaded_fingerprint = None
pdf_bytes = None
upload_error = None
if uploaded_pdf is not None:
	pdf_bytes = uploaded_pdf.getvalue()
	uploaded_fingerprint = hashlib.sha256(pdf_bytes).hexdigest()
	upload_event_id = uploaded_pdf.file_id
	if upload_event_id != st.session_state["last_uploaded_file_id"]:
		try:
			new_conversation = create_conversation(
				uploaded_pdf.name,
				uploaded_fingerprint,
			)
		except Exception as error:
			logger.error(
				"Could not save uploaded PDF conversation (%s).",
				type(error).__name__,
			)
			upload_error = "The PDF was uploaded, but its conversation could not be saved locally."
		else:
			st.session_state["last_uploaded_file_id"] = upload_event_id
			st.session_state["selected_conversation_id"] = new_conversation[
				"conversation_id"
			]

try:
	conversations = list_conversations()
except Exception as error:
	logger.error("Could not load saved conversations (%s).", type(error).__name__)
	conversations = []
	st.error("Saved conversation history could not be loaded from the local database.")

conversation_options = {
	conversation["conversation_id"]: conversation
	for conversation in conversations
}
if st.session_state["selected_conversation_id"] not in conversation_options:
	st.session_state["selected_conversation_id"] = (
		conversations[0]["conversation_id"] if conversations else None
	)

with st.sidebar:
	st.markdown("### :material/history: Conversation history")
	if conversations:
		selected_conversation_id = st.selectbox(
			"Open a saved conversation",
			options=[item["conversation_id"] for item in conversations],
			format_func=lambda conversation_id: (
				f"{conversation_options[conversation_id]['filename']} · "
				f"{conversation_options[conversation_id]['updated_at'][:16].replace('T', ' ')}"
			),
			key="selected_conversation_id",
			label_visibility="collapsed",
		)
	else:
		st.caption("Your saved conversations will appear here.")
		selected_conversation_id = None
	if conversations and st.button(
		"Clear history",
		icon=":material/delete_sweep:",
		width="stretch",
		help="Permanently delete all saved conversations and transcripts.",
	):
		confirm_clear_history()

try:
	selected_conversation = (
		get_conversation(selected_conversation_id)
		if selected_conversation_id is not None
		else None
	)
except Exception as error:
	logger.error("Could not open saved conversation (%s).", type(error).__name__)
	st.error("The selected conversation could not be opened from local history.")
	selected_conversation = None
conversation_id = (
	selected_conversation["conversation_id"] if selected_conversation else None
)
document_is_available = (
	upload_error is None
	and uploaded_pdf is not None
	and selected_conversation is not None
	and uploaded_fingerprint == selected_conversation["fingerprint"]
)

text_chunks = None
embeddings = None
page_count = None
if upload_error is not None:
	st.error(upload_error)

if document_is_available and pdf_bytes is not None and uploaded_pdf is not None:
	document_status.badge(
		"Preparing document",
		icon=":material/pending:",
		color="blue",
	)
	st.sidebar.caption(f"Selected file: {uploaded_pdf.name}")
	try:
		with st.sidebar:
			with st.status("Preparing document...", expanded=True) as processing_status:
				def update_processing_status(message: str) -> None:
					processing_status.update(label=message, state="running", expanded=True)

				text_chunks, embeddings, page_count = prepare_document(
					uploaded_fingerprint,
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
	except Exception:
		st.error("The PDF could not be processed. Please try another file.")
		document_status.badge("Processing failed", icon=":material/error:", color="red")

	if text_chunks is not None:
		document_status.badge("Ready", icon=":material/check_circle:", color="green")
		with document_stats.container(border=True):
			st.write(selected_conversation["filename"])
			metric_pages, metric_chunks = st.columns(2)
			metric_pages.metric("Pages", page_count)
			metric_chunks.metric("Text chunks", len(text_chunks))
elif selected_conversation is not None:
	document_status.badge(
		"Transcript only",
		icon=":material/history:",
		color="violet",
	)
	document_stats.markdown(f"**{selected_conversation['filename']}**")
	st.sidebar.info(
		"Re-upload this PDF to ask follow-up questions. Previous answers remain available below."
	)
elif uploaded_pdf is None:
	document_stats.caption("Upload a PDF or choose a saved conversation.")

if document_is_available and text_chunks is not None:
	header_status.badge("Document ready", icon=":material/check_circle:", color="green")
elif selected_conversation is not None:
	header_status.badge("Viewing saved history", icon=":material/history:", color="violet")
elif uploaded_pdf is not None:
	header_status.badge("Upload not saved", icon=":material/error:", color="red")

st.space("medium")
st.subheader("Conversation", icon=":material/chat:")

if selected_conversation is None:
	with st.container(border=True):
		st.markdown("#### Your document workspace")
		st.write("Upload a PDF in the sidebar to prepare it for question answering.")
		st.caption("Text extraction, chunking, and embedding progress will appear in the document panel.")
else:
	st.caption(
		f"Conversation started {selected_conversation['created_at'][:16].replace('T', ' ')} · "
		f"PDF uploaded {selected_conversation['uploaded_at'][:16].replace('T', ' ')}"
	)

try:
	document_history = (
		get_messages(conversation_id) if conversation_id is not None else []
	)
except Exception as error:
	logger.error("Could not load conversation transcript (%s).", type(error).__name__)
	document_history = []
	st.error("This conversation's transcript could not be loaded from the local database.")

if selected_conversation is not None and not document_history:
	with st.container(border=True):
		st.markdown("#### Start with a question")
		if document_is_available and text_chunks is not None:
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
		else:
			st.caption("Re-upload the matching PDF to start asking questions.")

for message in document_history:
	with st.chat_message("user", avatar=":material/person:"):
		st.markdown(message["question"])
	with st.chat_message("assistant", avatar=":material/auto_awesome:"):
		st.markdown(message["answer"])

if document_is_available and text_chunks is not None and embeddings is not None:
	with st.form("question_form", clear_on_submit=True):
		question = st.text_input(
			"Your question",
			key="question_input",
			placeholder="Ask something about this PDF...",
		)
		show_retrieval_diagnostics = st.checkbox(
			"Show retrieval diagnostics",
			value=False,
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
						if show_retrieval_diagnostics:
							for rank, chunk in enumerate(retrieved_chunks, start=1):
								page_number = chunk.get("page_number")
								page_label = (
									f"PDF page {page_number}"
									if page_number is not None
									else "PDF page unknown"
								)
								similarity = chunk.get("similarity")
								preview = " ".join(chunk["text"].split())
								if preview:
									preview = (
										preview[: min(157, len(preview) - 1)] + "..."
									)
								st.caption(
									f"Rank {rank} · {page_label} · "
									f"cosine similarity {similarity:.4f}"
								)
								st.text(preview)

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
						try:
							save_message(conversation_id, question, answer)
						except Exception as error:
							logger.error(
								"Could not save generated conversation turn (%s).",
								type(error).__name__,
							)
							st.error(
								"The answer was generated, but it could not be saved to local history."
							)
			except ValueError as error:
				if "GEMINI_API_KEY is not set" in str(error):
					st.error("Gemini is not configured. Add GEMINI_API_KEY to your local .env file and restart the app.")
				else:
					st.error("The question could not be processed. Please check it and try again.")
			except GeminiUnavailableError as error:
				st.error(str(error))
			except RuntimeError:
				st.error("Gemini could not generate an answer. Check your API configuration or try again shortly.")
			except Exception:
				st.error("Something went wrong while searching the PDF. Please try again.")
elif selected_conversation is not None:
	st.info(
		"To continue this conversation, upload the matching PDF. "
		"Saved questions and answers remain available without the file."
	)
