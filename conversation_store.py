"""SQLite storage for uploaded documents and their conversation history."""

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
from typing import Generator
import uuid


DEFAULT_DATABASE_PATH = (
	Path(__file__).parent / "data" / "conversations.sqlite3"
)


@contextmanager
def _connect(
	database_path: str | Path,
) -> Generator[sqlite3.Connection, None, None]:
	path = Path(database_path)
	path.parent.mkdir(parents=True, exist_ok=True)
	connection = sqlite3.connect(path)
	connection.row_factory = sqlite3.Row
	connection.execute("PRAGMA foreign_keys = ON")
	try:
		yield connection
		connection.commit()
	except Exception:
		connection.rollback()
		raise
	finally:
		connection.close()


def _utc_timestamp() -> str:
	return datetime.now(timezone.utc).isoformat(timespec="seconds")


def initialize_database(
	database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> None:
	"""Create the local conversation tables when they do not yet exist."""
	with _connect(database_path) as connection:
		connection.execute(
			"""
			CREATE TABLE IF NOT EXISTS documents (
				document_id TEXT PRIMARY KEY,
				fingerprint TEXT NOT NULL,
				filename TEXT NOT NULL,
				uploaded_at TEXT NOT NULL
			)
			"""
		)
		connection.execute(
			"""
			CREATE TABLE IF NOT EXISTS conversations (
				conversation_id TEXT PRIMARY KEY,
				document_id TEXT NOT NULL REFERENCES documents(document_id),
				created_at TEXT NOT NULL,
				updated_at TEXT NOT NULL
			)
			"""
		)
		connection.execute(
			"""
			CREATE TABLE IF NOT EXISTS messages (
				message_id INTEGER PRIMARY KEY AUTOINCREMENT,
				conversation_id TEXT NOT NULL
					REFERENCES conversations(conversation_id) ON DELETE CASCADE,
				question TEXT NOT NULL,
				answer TEXT NOT NULL,
				created_at TEXT NOT NULL
			)
			"""
		)
		connection.execute(
			"""
			CREATE INDEX IF NOT EXISTS messages_conversation_order
			ON messages(conversation_id, message_id)
			"""
		)
		connection.execute(
			"""
			CREATE INDEX IF NOT EXISTS conversations_updated
			ON conversations(updated_at DESC)
			"""
		)


def create_conversation(
	filename: str,
	fingerprint: str,
	database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> dict[str, str]:
	"""Record one upload and create a fresh conversation for it."""
	if not filename.strip():
		raise ValueError("filename cannot be empty.")
	if len(fingerprint) != 64:
		raise ValueError("fingerprint must be a SHA-256 hex digest.")
	try:
		int(fingerprint, 16)
	except ValueError as error:
		raise ValueError("fingerprint must be a SHA-256 hex digest.") from error

	document_id = str(uuid.uuid4())
	conversation_id = str(uuid.uuid4())
	uploaded_at = _utc_timestamp()

	with _connect(database_path) as connection:
		connection.execute(
			"""
			INSERT INTO documents (document_id, fingerprint, filename, uploaded_at)
			VALUES (?, ?, ?, ?)
			""",
			(document_id, fingerprint, filename, uploaded_at),
		)
		connection.execute(
			"""
			INSERT INTO conversations (
				conversation_id, document_id, created_at, updated_at
			)
			VALUES (?, ?, ?, ?)
			""",
			(conversation_id, document_id, uploaded_at, uploaded_at),
		)

	return {
		"document_id": document_id,
		"conversation_id": conversation_id,
		"fingerprint": fingerprint,
		"filename": filename,
		"uploaded_at": uploaded_at,
		"created_at": uploaded_at,
		"updated_at": uploaded_at,
	}


def list_conversations(
	database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> list[dict[str, str | int]]:
	"""Return conversation and document labels, newest activity first."""
	with _connect(database_path) as connection:
		rows = connection.execute(
			"""
			SELECT
				c.conversation_id,
				c.document_id,
				c.created_at,
				c.updated_at,
				d.filename,
				d.fingerprint,
				d.uploaded_at,
				COUNT(m.message_id) AS message_count
			FROM conversations AS c
			JOIN documents AS d ON d.document_id = c.document_id
			LEFT JOIN messages AS m ON m.conversation_id = c.conversation_id
			GROUP BY c.conversation_id
			ORDER BY c.updated_at DESC, c.created_at DESC
			"""
		).fetchall()
	return [dict(row) for row in rows]


def get_conversation(
	conversation_id: str,
	database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> dict[str, str] | None:
	"""Return one conversation with its associated uploaded document."""
	with _connect(database_path) as connection:
		row = connection.execute(
			"""
			SELECT
				c.conversation_id,
				c.document_id,
				c.created_at,
				c.updated_at,
				d.filename,
				d.fingerprint,
				d.uploaded_at
			FROM conversations AS c
			JOIN documents AS d ON d.document_id = c.document_id
			WHERE c.conversation_id = ?
			""",
			(conversation_id,),
		).fetchone()
	return dict(row) if row is not None else None


def get_messages(
	conversation_id: str,
	database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> list[dict[str, str | int]]:
	"""Return the saved question-and-answer turns in their original order."""
	with _connect(database_path) as connection:
		rows = connection.execute(
			"""
			SELECT message_id, question, answer, created_at
			FROM messages
			WHERE conversation_id = ?
			ORDER BY message_id
			""",
			(conversation_id,),
		).fetchall()
	return [dict(row) for row in rows]


def save_message(
	conversation_id: str,
	question: str,
	answer: str,
	database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> None:
	"""Save one successful answer and update its conversation activity time."""
	if not question.strip():
		raise ValueError("question cannot be empty.")
	if not answer.strip():
		raise ValueError("answer cannot be empty.")

	created_at = _utc_timestamp()
	with _connect(database_path) as connection:
		result = connection.execute(
			"""
			UPDATE conversations
			SET updated_at = ?
			WHERE conversation_id = ?
			""",
			(created_at, conversation_id),
		)
		if result.rowcount == 0:
			raise ValueError("conversation does not exist.")
		connection.execute(
			"""
			INSERT INTO messages (conversation_id, question, answer, created_at)
			VALUES (?, ?, ?, ?)
			""",
			(conversation_id, question, answer, created_at),
		)


def clear_history(database_path: str | Path = DEFAULT_DATABASE_PATH) -> int:
	"""Delete all saved conversations, messages, and their document records."""
	with _connect(database_path) as connection:
		conversation_count = connection.execute(
			"SELECT COUNT(*) FROM conversations"
		).fetchone()[0]
		connection.execute("DELETE FROM conversations")
		connection.execute("DELETE FROM documents")
	return int(conversation_count)
