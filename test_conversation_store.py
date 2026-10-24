import tempfile
import unittest
from pathlib import Path

from conversation_store import (
	create_conversation,
	clear_history,
	get_conversation,
	get_messages,
	initialize_database,
	list_conversations,
	save_message,
)


FINGERPRINT = "a" * 64


class ConversationStoreTests(unittest.TestCase):
	def setUp(self) -> None:
		self.temp_dir = tempfile.TemporaryDirectory()
		self.addCleanup(self.temp_dir.cleanup)
		self.database_path = Path(self.temp_dir.name) / "test.sqlite3"
		initialize_database(self.database_path)

	def test_each_upload_creates_its_own_conversation(self) -> None:
		first = create_conversation("report.pdf", FINGERPRINT, self.database_path)
		second = create_conversation("report.pdf", FINGERPRINT, self.database_path)

		self.assertNotEqual(first["conversation_id"], second["conversation_id"])
		self.assertNotEqual(first["document_id"], second["document_id"])
		self.assertEqual(first["fingerprint"], FINGERPRINT)

		conversations = list_conversations(self.database_path)
		self.assertEqual(len(conversations), 2)
		self.assertEqual(
			{item["conversation_id"] for item in conversations},
			{first["conversation_id"], second["conversation_id"]},
		)

	def test_saved_messages_and_document_metadata_survive_reopening(self) -> None:
		conversation = create_conversation(
			"research.pdf",
			FINGERPRINT,
			self.database_path,
		)
		save_message(
			conversation["conversation_id"],
			"What is the main finding?",
			"The main finding is recorded.",
			self.database_path,
		)

		reopened = get_conversation(
			conversation["conversation_id"],
			self.database_path,
		)
		messages = get_messages(conversation["conversation_id"], self.database_path)

		self.assertIsNotNone(reopened)
		assert reopened is not None
		self.assertEqual(reopened["filename"], "research.pdf")
		self.assertEqual(reopened["fingerprint"], FINGERPRINT)
		self.assertIn("T", reopened["uploaded_at"])
		self.assertIn("T", reopened["created_at"])
		self.assertIn("T", reopened["updated_at"])
		self.assertEqual(len(messages), 1)
		self.assertEqual(messages[0]["question"], "What is the main finding?")
		self.assertEqual(messages[0]["answer"], "The main finding is recorded.")

	def test_conversation_list_includes_message_count_and_document_name(self) -> None:
		conversation = create_conversation(
			"notes.pdf",
			FINGERPRINT,
			self.database_path,
		)
		save_message(
			conversation["conversation_id"],
			"Question",
			"Answer",
			self.database_path,
		)

		item = list_conversations(self.database_path)[0]

		self.assertEqual(item["filename"], "notes.pdf")
		self.assertEqual(item["message_count"], 1)

	def test_invalid_upload_data_and_missing_conversations_are_rejected(self) -> None:
		with self.assertRaisesRegex(ValueError, "filename"):
			create_conversation(" ", FINGERPRINT, self.database_path)
		with self.assertRaisesRegex(ValueError, "SHA-256"):
			create_conversation("report.pdf", "not-a-fingerprint", self.database_path)
		with self.assertRaisesRegex(ValueError, "does not exist"):
			save_message("missing-conversation", "Question", "Answer", self.database_path)
		with self.assertRaisesRegex(ValueError, "question"):
			save_message("missing-conversation", " ", "Answer", self.database_path)

	def test_initialize_database_is_safe_to_call_more_than_once(self) -> None:
		initialize_database(self.database_path)
		conversation = create_conversation(
			"repeat.pdf",
			FINGERPRINT,
			self.database_path,
		)

		self.assertEqual(len(list_conversations(self.database_path)), 1)
		self.assertIsNotNone(
			get_conversation(conversation["conversation_id"], self.database_path)
		)

	def test_clear_history_deletes_all_conversations_messages_and_documents(self) -> None:
		first = create_conversation("first.pdf", FINGERPRINT, self.database_path)
		second = create_conversation("second.pdf", "b" * 64, self.database_path)
		save_message(first["conversation_id"], "Question one", "Answer one", self.database_path)
		save_message(second["conversation_id"], "Question two", "Answer two", self.database_path)

		deleted_count = clear_history(self.database_path)

		self.assertEqual(deleted_count, 2)
		self.assertEqual(list_conversations(self.database_path), [])
		self.assertEqual(get_messages(first["conversation_id"], self.database_path), [])
		self.assertIsNone(get_conversation(first["conversation_id"], self.database_path))
		self.assertIsNone(get_conversation(second["conversation_id"], self.database_path))
		self.assertEqual(clear_history(self.database_path), 0)


if __name__ == "__main__":
	unittest.main()
