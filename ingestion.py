"""Safe, bounded extraction of plain text from supported document formats."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

MAX_DOCUMENT_BYTES = 10 * 1024 * 1024


def extract_text(name: str, payload: bytes) -> str:
    """Extract text from TXT, PDF, or DOCX bytes; uploaded data is never executed."""
    suffix = Path(name).suffix.casefold()
    if suffix not in {".txt", ".pdf", ".docx"}:
        raise ValueError("Supported document formats are TXT, PDF, and DOCX")
    if len(payload) > MAX_DOCUMENT_BYTES:
        raise ValueError("Document exceeds the 10 MiB upload limit")
    if suffix == ".txt":
        text = payload.decode("utf-8-sig")
    elif suffix == ".pdf":
        try:
            from pypdf import PdfReader
            reader = PdfReader(BytesIO(payload), strict=True)
            if len(reader.pages) > 500:
                raise ValueError("PDF exceeds the 500 page limit")
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        except ValueError:
            raise
        except Exception as error:
            raise ValueError(f"Could not read PDF: {error}") from error
    else:
        try:
            from docx import Document
            document = Document(BytesIO(payload))
            text = "\n".join(paragraph.text for paragraph in document.paragraphs)
        except Exception as error:
            raise ValueError(f"Could not read DOCX: {error}") from error
    if not text.strip():
        raise ValueError("Document contains no extractable text")
    return text
