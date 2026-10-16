from __future__ import annotations

import re
import math
import threading
from typing import List

import numpy as np

from config import logger

_BACKEND = None          
_st_model = None         
_lock = threading.Lock()


def _try_load_st():
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
        logger.info("embeddings: 使用 sentence-transformers 后端 (bge-small-zh-v1.5)")
        return model
    except Exception as e:
        logger.warning("embeddings: sentence-transformers 不可用，回退 TF-IDF。原因: %s", e)
        return None


def _ensure_backend():
    global _BACKEND, _st_model
    if _BACKEND is not None:
        return
    with _lock:
        if _BACKEND is not None:
            return
        model = _try_load_st()
        if model is not None:
            _st_model = model
            _BACKEND = "st"
        else:
            _BACKEND = "tfidf"


def backend_name() -> str:
    _ensure_backend()
    return _BACKEND


_token_re = re.compile(r"[一-鿿]|[a-zA-Z0-9]+")


def _tokenize(text: str) -> List[str]:
    return _token_re.findall((text or "").lower())


class _TfidfState:
    vocab: dict = {}
    idf: np.ndarray = np.zeros(0, dtype=np.float32)
    doc_count: int = 0
    df: dict = {}


_tfidf = _TfidfState()


def _tfidf_fit_transform(texts: List[str]) -> np.ndarray:
    for t in texts:
        seen = set(_tokenize(t))
        for tok in seen:
            if tok not in _tfidf.vocab:
                _tfidf.vocab[tok] = len(_tfidf.vocab)
            _tfidf.df[tok] = _tfidf.df.get(tok, 0) + 1
        _tfidf.doc_count += 1

    dim = len(_tfidf.vocab)
    idf = np.zeros(dim, dtype=np.float32)
    for tok, idx in _tfidf.vocab.items():
        df = _tfidf.df.get(tok, 0)
        idf[idx] = math.log((1 + _tfidf.doc_count) / (1 + df)) + 1.0
    _tfidf.idf = idf
    return _tfidf_transform(texts)


def _tfidf_transform(texts: List[str]) -> np.ndarray:
    dim = len(_tfidf.vocab)
    mat = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        toks = _tokenize(t)
        if not toks:
            continue
        for tok in toks:
            idx = _tfidf.vocab.get(tok)
            if idx is not None:
                mat[i, idx] += 1.0
        mat[i] /= max(len(toks), 1)
        if _tfidf.idf.shape[0] == dim:
            mat[i] *= _tfidf.idf
    return _l2_normalize(mat)


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (mat / norms).astype(np.float32)


def embed(texts, is_query: bool = False) -> np.ndarray:
    if isinstance(texts, str):
        texts = [texts]
    _ensure_backend()

    if _BACKEND == "st":
        vecs = _st_model.encode(
            list(texts), normalize_embeddings=True, convert_to_numpy=True
        )
        return vecs.astype(np.float32)

    if is_query and len(_tfidf.vocab) > 0:
        return _tfidf_transform(list(texts))
    return _tfidf_fit_transform(list(texts))


if __name__ == "__main__":
    docs = ["AI 职业发展前景如何", "Python 课程包含哪些内容", "今天天气怎么样"]
    m = embed(docs)
    q = embed("人工智能就业", is_query=True)
    sims = (m @ q.T).ravel()
    print("backend =", backend_name(), "dim =", m.shape[1])
    for d, s in sorted(zip(docs, sims), key=lambda x: -x[1]):
        print(f"  {s:.3f}  {d}")
