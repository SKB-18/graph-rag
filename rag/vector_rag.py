import math
import re
from typing import List, Dict, Any, Tuple


class NaiveVectorRAG:
    """
    Standard Naive Vector / Semantic Retrieval Baseline.
    Indexes text chunks using TF-IDF / term-frequency cosine vector space,
    retrieving top-k chunks matching the surface text query.
    """
    def __init__(self):
        self.chunks: List[str] = []
        self.vocab: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.chunk_vectors: List[Dict[str, float]] = []

    def _tokenize(self, text: str) -> List[str]:
        return [w.lower() for w in re.findall(r"\b[a-zA-Z0-9_]+\b", text)]

    def index(self, documents: List[str]):
        """Splits documents into sentence chunks and indexes TF-IDF vectors."""
        self.chunks = []
        for doc in documents:
            for s in re.split(r"[.\n;]", doc):
                s_clean = s.strip()
                if len(s_clean) > 5:
                    self.chunks.append(s_clean)

        num_docs = len(self.chunks)
        if num_docs == 0:
            return

        doc_frequencies: Dict[str, int] = {}
        tokenized_chunks = []

        for chunk in self.chunks:
            tokens = set(self._tokenize(chunk))
            tokenized_chunks.append(self._tokenize(chunk))
            for tok in tokens:
                doc_frequencies[tok] = doc_frequencies.get(tok, 0) + 1

        # Calculate IDF
        self.idf = {
            tok: math.log((num_docs + 1) / (df + 1)) + 1.0
            for tok, df in doc_frequencies.items()
        }

        # Calculate TF-IDF vectors for each chunk
        self.chunk_vectors = []
        for tokens in tokenized_chunks:
            tf: Dict[str, float] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1.0
            
            vec: Dict[str, float] = {
                t: (count / len(tokens)) * self.idf.get(t, 1.0)
                for t, count in tf.items()
            }
            # L2 normalize
            norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
            self.chunk_vectors.append({t: v / norm for t, v in vec.items()})

    def query(self, query_text: str, top_k: int = 2) -> List[Dict[str, Any]]:
        """Returns the top_k most similar chunks by cosine similarity."""
        tokens = self._tokenize(query_text)
        if not tokens or not self.chunk_vectors:
            return []

        # Vectorize query
        q_tf: Dict[str, float] = {}
        for t in tokens:
            q_tf[t] = q_tf.get(t, 0) + 1.0
            
        q_vec = {
            t: (cnt / len(tokens)) * self.idf.get(t, 1.0)
            for t, cnt in q_tf.items()
        }
        q_norm = math.sqrt(sum(v * v for v in q_vec.values())) or 1.0
        normalized_q = {t: v / q_norm for t, v in q_vec.items()}

        # Compute cosine scores
        scores: List[Tuple[float, int]] = []
        for idx, c_vec in enumerate(self.chunk_vectors):
            dot = sum(c_vec.get(t, 0.0) * w for t, w in normalized_q.items())
            scores.append((dot, idx))

        scores.sort(key=lambda x: x[0], reverse=True)
        
        results = []
        for score, idx in scores[:top_k]:
            results.append({
                "score": round(score, 4),
                "chunk": self.chunks[idx]
            })
            
        return results
