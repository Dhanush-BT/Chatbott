"""
document_extractor.py

Standalone extractor for PDF and DOCX files.
– Parses the supplied documents
– Cleans the text
– Emits the full extraction as JSON to stdout
"""

import os
import sys
import json
import logging
import re
from pathlib import Path
from typing import List, Dict

try:
    import fitz  # PyMuPDF
except ImportError:                                 # pragma: no cover
    print("PyMuPDF (fitz) not installed.  Install with:  pip install pymupdf")
    sys.exit(1)

try:
    from docx import Document
except ImportError:                                 # pragma: no cover
    print("python-docx not installed.  Install with:  pip install python-docx")
    sys.exit(1)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("extractor")


SUPPORTED_EXT = {".pdf", ".docx"}


# --------------------------------------------------------------------------- #
#  Core extractor
# --------------------------------------------------------------------------- #
class DocumentExtractor:
    """Extract text from PDF and DOCX files."""

    def __init__(self) -> None:
        self.results: List[Dict] = []

    # -------- public API --------------------------------------------------- #
    def extract_from_paths(self, paths: List[str]) -> List[Dict]:
        """Extract text from all paths passed in."""
        for path in paths:
            if not os.path.isfile(path):
                logger.warning("Not a file: %s", path)
                continue

            ext = Path(path).suffix.lower()
            if ext == ".pdf":
                self.results.extend(self._extract_pdf(path))
            elif ext == ".docx":
                self.results.extend(self._extract_docx(path))
            else:
                logger.warning("Unsupported extension: %s", path)

        return self.results

    # -------- helpers ------------------------------------------------------ #
    def _extract_pdf(self, pdf_path: str) -> List[Dict]:
        segments: List[Dict] = []
        try:
            doc = fitz.open(pdf_path)
            for page_idx in range(len(doc)):
                page = doc[page_idx]
                text = page.get_text("text").strip()
                if text:
                    segments.append(
                        {
                            "filename": os.path.basename(pdf_path),
                            "page_number": page_idx + 1,
                            "chunk_id": 0,
                            "text": self._clean(text),
                            "file_path": pdf_path,
                            "document_type": "pdf",
                        }
                    )
            doc.close()
            logger.info("PDF extracted: %s (%d pages)", pdf_path, len(segments))
        except Exception as exc:                              # pragma: no cover
            logger.error("Failed PDF extract %s :: %s", pdf_path, exc)
        return segments

    def _extract_docx(self, docx_path: str) -> List[Dict]:
        try:
            doc = Document(docx_path)
            lines = [
                p.text.strip()
                if not p.style.name.startswith("Heading")
                else f"\n{p.text.strip()}\n"
                for p in doc.paragraphs
                if p.text.strip()
            ]
            text = self._clean("\n\n".join(lines))
            if text:
                return [
                    {
                        "filename": os.path.basename(docx_path),
                        "page_number": 1,
                        "chunk_id": 0,
                        "text": text,
                        "file_path": docx_path,
                        "document_type": "docx",
                    }
                ]
            logger.warning("No text in DOCX: %s", docx_path)
        except Exception as exc:                              # pragma: no cover
            logger.error("Failed DOCX extract %s :: %s", docx_path, exc)
        return []

    @staticmethod
    def _clean(text: str) -> str:
        text = re.sub(r"\n\s*\n", "\n\n", text)   # collapse blank lines
        text = re.sub(r"[ ]{2,}", " ", text)      # collapse multiple spaces
        return text.strip()


# --------------------------------------------------------------------------- #
#  CLI helper
# --------------------------------------------------------------------------- #
def collect_paths(argv: List[str]) -> List[str]:
    """
    Accepts a mixture of file and directory paths from the command line,
    returning all supported-document paths found.
    """
    if not argv:
        print("Usage:  python document_extractor.py <file_or_dir> [more ...]")
        sys.exit(1)

    discovered: List[str] = []
    for item in argv:
        p = Path(item).expanduser().resolve()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXT:
            discovered.append(str(p))
        elif p.is_dir():
            discovered.extend(
                str(fp) for fp in p.rglob("*") if fp.suffix.lower() in SUPPORTED_EXT
            )
        else:
            logger.warning("Skipping unsupported path: %s", item)

    if not discovered:
        logger.error("No supported documents found.")
        sys.exit(1)

    return discovered


# --------------------------------------------------------------------------- #
#  entry-point
# --------------------------------------------------------------------------- #
def main() -> None:
    paths = collect_paths(sys.argv[1:])

    extractor = DocumentExtractor()
    segments = extractor.extract_from_paths(paths)

    # Output the result as pretty-printed JSON
    print(json.dumps({"segments": segments}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
