"""Section-first chunking. Sizes are in words (a rough proxy for tokens; not measured)."""

from scoper.ingestion.models import REQUIRED_SECTIONS, BriefDoc
from scoper.retrieval.store import Chunk


def _pack(paragraphs: list[str], max_words: int, overlap: int) -> list[str]:
    out, cur, cur_n = [], [], 0
    for p in paragraphs:
        words = p.split()
        if len(words) > max_words:  # one huge paragraph: sliding window with overlap
            if cur:
                out.append("\n\n".join(cur))
                cur, cur_n = [], 0
            step = max_words - overlap
            for i in range(0, len(words), step):
                out.append(" ".join(words[i : i + max_words]))
                if i + max_words >= len(words):
                    break
            continue
        if cur_n + len(words) > max_words and cur:
            out.append("\n\n".join(cur))
            cur, cur_n = [], 0
        cur.append(p)
        cur_n += len(words)
    if cur:
        out.append("\n\n".join(cur))
    return out


def chunk_brief(doc: BriefDoc, max_words: int = 200, overlap: int = 30) -> list[Chunk]:
    meta = doc.meta()
    chunks = []
    for section in REQUIRED_SECTIONS:
        paragraphs = [p.strip() for p in doc.sections[section].split("\n\n") if p.strip()]
        for n, text in enumerate(_pack(paragraphs, max_words, overlap)):
            chunks.append(Chunk(f"{doc.id}:{section}:{n}", doc.id, section, n, text, dict(meta)))
    return chunks


def embed_text(chunk: Chunk) -> str:
    """Text actually embedded: title and section prefix give each chunk its context."""
    return f"{chunk.meta.get('title', '')} | {chunk.section}\n{chunk.text}"
