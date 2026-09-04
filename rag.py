"""
rag.py — lightweight knowledge-grounding layer for Hasha.

This is intentionally dependency-light (no vector DB, no external embedding
API) so it runs anywhere. It chunks uploaded material into paragraphs and
retrieves the most relevant chunks for a given concept using TF-IDF cosine
similarity. For a production submission, swap `TfidfRetriever` for a real
embedding model (OpenAI/Voyage/sentence-transformers) + a vector store
(Chroma/FAISS/Qdrant) — the interface (`chunk()` / `retrieve()`) stays the same.
"""

import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def chunk_material(text: str, min_len: int = 40) -> list[str]:
    """Split raw material into paragraph-level chunks.

    Splits on blank lines first (natural paragraph/section boundaries in
    most uploaded notes/textbook exports). Falls back to sentence grouping
    if the material has no blank-line structure (e.g. a single dense block
    of text pulled from a PDF).
    """
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    paragraphs = [p for p in paragraphs if len(p) >= min_len]

    if len(paragraphs) >= 3:
        return paragraphs

    # Fallback: no clear paragraph breaks — group sentences into ~3-sentence chunks
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks, buf = [], []
    for s in sentences:
        buf.append(s)
        if len(buf) >= 3:
            chunks.append(" ".join(buf))
            buf = []
    if buf:
        chunks.append(" ".join(buf))
    return [c for c in chunks if len(c) >= min_len]


class TfidfRetriever:
    """Fits a TF-IDF index over material chunks and retrieves top-k matches
    for a query (typically a concept title, e.g. "Ohm's Law")."""

    def __init__(self, chunks: list[str]):
        self.chunks = chunks
        self.vectorizer = None
        self.matrix = None
        if chunks:
            self.vectorizer = TfidfVectorizer(stop_words="english")
            self.matrix = self.vectorizer.fit_transform(chunks)

    def retrieve(self, query: str, top_k: int = 2, min_score: float = 0.05) -> list[str]:
        if not self.chunks or self.vectorizer is None:
            return []
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix).flatten()
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        results = [self.chunks[i] for i in ranked[:top_k] if scores[i] >= min_score]
        return results
