import tempfile
import unittest
from pathlib import Path

import fitz

from rag import chunk_text, extract_text_from_pdf


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


if __name__ == "__main__":
    unittest.main()
