"""Turn a filing PDF into page-level chunks that keep their source metadata."""

import re
from dataclasses import asdict, dataclass
from pathlib import Path

import pymupdf

MAX_CHARS = 2000
OVERLAP = 200


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    company: str
    fy: str
    doc_type: str
    page: int
    text: str

    def payload(self) -> dict:
        return asdict(self)


# Some reports (TCS's) embed fonts whose digits extract as the Coptic letters U+03EC..U+03F5
# in order (Ϭ=0 ... ϵ=9), and the "ffi" ligature as ĸ. English filings never use these, so
# mapping them back is safe, and it makes the figures searchable.
GLYPH_FIXES = str.maketrans({**{chr(0x03EC + d): str(d) for d in range(10)}, "ĸ": "ffi"})


def clean(text: str) -> str:
    text = text.translate(GLYPH_FIXES)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_page(text: str, max_chars: int = MAX_CHARS, overlap: int = OVERLAP) -> list[str]:
    """Split one page's text into windows; most pages fit in one."""
    if len(text) <= max_chars:
        return [text] if text else []
    parts, start = [], 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            cut = text.rfind("\n", start + max_chars // 2, end)
            end = cut if cut != -1 else end
        parts.append(text[start:end].strip())
        if end >= len(text):
            break
        start = end - overlap
    return [p for p in parts if p]


def parse_pdf(path: Path, company: str, fy: str, doc_type: str = "annual_report") -> list[Chunk]:
    doc_id = f"{company}_{fy}_{doc_type}"
    chunks = []
    with pymupdf.open(path) as pdf:
        for page_no, page in enumerate(pdf, start=1):
            for i, part in enumerate(split_page(clean(page.get_text()))):
                chunks.append(
                    Chunk(f"{doc_id}_p{page_no}_{i}", doc_id, company, fy, doc_type, page_no, part)
                )
    return chunks
