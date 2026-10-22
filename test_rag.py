import tempfile
import traceback
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import fitz
import numpy as np

from rag import (
    chunk_text,
    extract_text_from_pdf,
    generate_answer,
    generate_embeddings,
    semantic_search,
)


class ExtractTextFromPdfTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.temp_path = Path(self.temp_dir.name)

    def _create_pdf(self, filename: str, page_texts: list[str]) -> Path:
        pdf_path = self.temp_path / filename
        document = fitz.open()
        for text in page_texts:
            page = document.new_page()
            if text:
                page.insert_text((72, 72), text)
        document.save(pdf_path)
        document.close()
        return pdf_path

    def test_extracts_normalized_text_from_each_page(self) -> None:
        pdf_path = self._create_pdf("valid.pdf", ["Hello   world", "Second   page"])

        pages = extract_text_from_pdf(str(pdf_path))

        self.assertEqual(
            pages,
            [
                {"page_number": 1, "text": "Hello world"},
                {"page_number": 2, "text": "Second page"},
            ],
        )

    def test_missing_pdf_raises_clear_value_error(self) -> None:
        missing_path = self.temp_path / "missing.pdf"

        with self.assertRaisesRegex(ValueError, "Unable to read PDF"):
            extract_text_from_pdf(str(missing_path))

    def test_invalid_pdf_raises_clear_value_error(self) -> None:
        invalid_path = self.temp_path / "invalid.pdf"
        invalid_path.write_bytes(b"not a valid PDF")

        with self.assertRaisesRegex(ValueError, "Unable to read PDF"):
            extract_text_from_pdf(str(invalid_path))

    def test_pdf_without_extractable_text_raises_value_error(self) -> None:
        pdf_path = self._create_pdf("textless.pdf", [""])

        with self.assertRaisesRegex(ValueError, "no extractable text"):
            extract_text_from_pdf(str(pdf_path))


class ChunkTextTests(unittest.TestCase):
    def test_splits_text_into_overlapping_chunks(self) -> None:
        chunks = chunk_text("abcdefghij", chunk_size=4, overlap=1)

        self.assertEqual(chunks, ["abcd", "defg", "ghij"])
        self.assertEqual(chunks[0][-1], chunks[1][0])
        self.assertEqual(chunks[1][-1], chunks[2][0])

    def test_empty_or_whitespace_input_returns_no_chunks(self) -> None:
        self.assertEqual(chunk_text(""), [])
        self.assertEqual(chunk_text("   \n\t"), [])

    def test_invalid_chunk_settings_raise_value_error(self) -> None:
        invalid_settings = [(0, 0), (-1, 0), (4, -1), (4, 4)]

        for chunk_size, overlap in invalid_settings:
            with self.subTest(chunk_size=chunk_size, overlap=overlap):
                with self.assertRaises(ValueError):
                    chunk_text("some text", chunk_size, overlap)


class EmbeddingTests(unittest.TestCase):
    @patch("rag._get_embedding_model")
    def test_generates_embeddings_from_text_and_metadata_records(self, get_model) -> None:
        model = get_model.return_value
        model.encode.return_value = np.array([[1.0, 0.0], [0.0, 1.0]])
        chunks = [
            {"text": "first page chunk", "page_number": 1},
            "second chunk",
        ]

        embeddings = generate_embeddings(chunks)

        np.testing.assert_array_equal(
            embeddings,
            np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        )
        model.encode.assert_called_once_with(
            ["first page chunk", "second chunk"],
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    @patch("rag._get_embedding_model")
    def test_empty_chunk_list_returns_empty_embeddings_without_loading_model(
        self, get_model
    ) -> None:
        embeddings = generate_embeddings([])

        self.assertEqual(embeddings.shape, (0, 0))
        get_model.assert_not_called()

    def test_invalid_chunk_records_raise_clear_errors(self) -> None:
        with self.assertRaises(TypeError):
            generate_embeddings([123])
        with self.assertRaisesRegex(ValueError, "text"):
            generate_embeddings([{"page_number": 1}])


class SemanticSearchTests(unittest.TestCase):
    @patch("rag._get_embedding_model")
    def test_ranks_by_cosine_similarity_and_preserves_metadata(self, get_model) -> None:
        get_model.return_value.encode.return_value = np.array([[1.0, 0.0]])
        chunks = [
            {"text": "best match", "page_number": 7, "section": "intro"},
            {"text": "weaker match", "page_number": 3},
            {"text": "zero vector", "page_number": 9},
        ]
        embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])

        results = semantic_search("a query", chunks, embeddings, top_k=2)

        self.assertEqual([result["text"] for result in results], ["best match", "weaker match"])
        self.assertEqual(results[0]["page_number"], 7)
        self.assertEqual(results[0]["section"], "intro")
        self.assertAlmostEqual(results[0]["similarity"], 1.0)
        self.assertAlmostEqual(results[1]["similarity"], 0.0)

    @patch("rag._get_embedding_model")
    def test_zero_query_vector_returns_finite_zero_scores(self, get_model) -> None:
        get_model.return_value.encode.return_value = np.array([[0.0, 0.0]])

        results = semantic_search("query", ["chunk"], [[1.0, 0.0]])

        self.assertEqual(results[0]["similarity"], 0.0)
        self.assertTrue(np.isfinite(results[0]["similarity"]))

    @patch("rag._get_embedding_model")
    def test_empty_query_or_chunks_returns_no_results_without_encoding(self, get_model) -> None:
        self.assertEqual(semantic_search("  ", ["chunk"], [[1.0, 0.0]]), [])
        self.assertEqual(semantic_search("query", [], np.empty((0, 0))), [])
        get_model.assert_not_called()

    def test_invalid_top_k_values_raise_value_error(self) -> None:
        for top_k in (0, -1, 1.5, True):
            with self.subTest(top_k=top_k):
                with self.assertRaises(ValueError):
                    semantic_search("query", ["chunk"], [[1.0, 0.0]], top_k)

    @patch("rag._get_embedding_model")
    def test_embedding_count_and_dimensions_are_validated(self, get_model) -> None:
        with self.assertRaisesRegex(ValueError, "one vector per text chunk"):
            semantic_search("query", ["first", "second"], [[1.0, 0.0]])

        get_model.return_value.encode.return_value = np.array([[1.0, 0.0, 0.0]])
        with self.assertRaisesRegex(ValueError, "dimension"):
            semantic_search("query", ["chunk"], [[1.0, 0.0]])


class GenerateAnswerTests(unittest.TestCase):
    def test_successful_response_uses_context_pages_and_configured_model(self) -> None:
        with (
            patch.dict(
                "os.environ",
                {"GEMINI_API_KEY": "test-key", "GEMINI_MODEL": "gemini-test"},
            ),
            patch("rag.load_dotenv") as load_dotenv,
            patch("rag.genai.Client") as client_class,
        ):
            client = client_class.return_value
            client.models.generate_content.return_value = SimpleNamespace(
                text="  The document says 42.  "
            )

            answer = generate_answer(
                "What is the answer?",
                [
                    {"text": "The answer is 42.", "page_number": 6},
                    "Another relevant passage.",
                ],
            )

        self.assertEqual(answer, "The document says 42.")
        load_dotenv.assert_called_once_with(override=False)
        client_class.assert_called_once_with(api_key="test-key")
        call = client.models.generate_content.call_args.kwargs
        self.assertEqual(call["model"], "gemini-test")
        prompt = call["contents"]
        self.assertIn("[Source page: 6]\nThe answer is 42.", prompt)
        self.assertIn("Another relevant passage.", prompt)
        self.assertIn("What is the answer?", prompt)
        self.assertIn("using only the supplied PDF context", prompt)
        self.assertIn("Do not invent facts", prompt)
        self.assertIn("clearly say so", prompt)
        self.assertNotIn("Source page: 2", prompt)

    def test_uses_default_model_when_environment_value_is_empty(self) -> None:
        with (
            patch.dict("os.environ", {"GEMINI_API_KEY": "test-key", "GEMINI_MODEL": ""}),
            patch("rag.load_dotenv"),
            patch("rag.genai.Client") as client_class,
        ):
            client_class.return_value.models.generate_content.return_value = (
                SimpleNamespace(text="Answer")
            )

            generate_answer("Question?", ["Context."])

        call = client_class.return_value.models.generate_content.call_args.kwargs
        self.assertEqual(call["model"], "gemini-2.5-flash")
        self.assertNotIn("Source page:", call["contents"])

    def test_empty_question_or_context_raises_value_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "question"):
            generate_answer("  ", ["Context"])
        with self.assertRaisesRegex(ValueError, "non-empty text"):
            generate_answer("Question", [])
        with self.assertRaisesRegex(ValueError, "non-empty text"):
            generate_answer("Question", ["  "])

    def test_missing_api_key_raises_clear_error_without_calling_client(self) -> None:
        with (
            patch.dict("os.environ", {"GEMINI_API_KEY": ""}),
            patch("rag.load_dotenv"),
            patch("rag.genai.Client") as client_class,
        ):
            with self.assertRaisesRegex(ValueError, "GEMINI_API_KEY is not set"):
                generate_answer("Question", ["Context"])

        client_class.assert_not_called()

    def test_api_failure_is_reported_without_exposing_secrets(self) -> None:
        secret = "do-not-include-this-key"
        with (
            patch.dict("os.environ", {"GEMINI_API_KEY": secret}),
            patch("rag.load_dotenv"),
            patch("rag.genai.Client", side_effect=RuntimeError(secret)),
        ):
            with self.assertRaisesRegex(RuntimeError, "Gemini API request failed") as error:
                generate_answer("Question", ["Context"])

        self.assertNotIn(secret, str(error.exception))
        self.assertNotIn(secret, "".join(traceback.format_exception(error.exception)))

    def test_empty_api_response_raises_clear_error(self) -> None:
        with (
            patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}),
            patch("rag.load_dotenv"),
            patch("rag.genai.Client") as client_class,
        ):
            client_class.return_value.models.generate_content.return_value = (
                SimpleNamespace(text=" ")
            )

            with self.assertRaisesRegex(RuntimeError, "empty answer"):
                generate_answer("Question", ["Context"])


if __name__ == "__main__":
    unittest.main()
