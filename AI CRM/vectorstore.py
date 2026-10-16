from __future__ import annotations

import json
import threading
import uuid
from typing import List, Dict, Optional

import numpy as np

from config import VEC_DIR, logger, RAG_TOP_K
from embeddings import embed


class LocalVectorStore:
    def __init__(self, name: str):
        self.name = name
        self.vec_path = VEC_DIR / f"{name}.npy"
        self.meta_path = VEC_DIR / f"{name}.json"
        self._lock = threading.Lock()
        self._vectors: Optional[np.ndarray] = None   
        self._docs: List[Dict] = []                  
        self._load()

    def _load(self):
        if self.meta_path.exists():
            with open(self.meta_path, "r", encoding="utf-8") as f:
                self._docs = json.load(f)
        if self.vec_path.exists():
            self._vectors = np.load(self.vec_path).astype(np.float32)
        logger.info("vectorstore[%s]: 载入 %d 条文档", self.name, len(self._docs))

    def _persist(self):
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self._docs, f, ensure_ascii=False, indent=2)
        if self._vectors is not None:
            np.save(self.vec_path, self._vectors)

    def add(self, documents: List[str], metadatas: Optional[List[Dict]] = None) -> List[str]:
        if not documents:
            return []
        metadatas = metadatas or [{} for _ in documents]
        with self._lock:
            new_vecs = embed(documents)  
            ids = []
            for text, meta in zip(documents, metadatas):
                doc_id = str(uuid.uuid4())
                ids.append(doc_id)
                self._docs.append({"id": doc_id, "text": text, "meta": meta})
            if self._vectors is None:
                self._vectors = new_vecs
            else:
            
                self._vectors = _vstack_aligned(self._vectors, new_vecs)
            self._persist()
            logger.info("vectorstore[%s]: 新增 %d 条，共 %d 条",
                        self.name, len(documents), len(self._docs))
            return ids

    def update(self, doc_id: str, text: str, meta: Optional[Dict] = None) -> bool:
        with self._lock:
            for i, d in enumerate(self._docs):
                if d["id"] == doc_id:
                    d["text"] = text
                    if meta is not None:
                        d["meta"] = meta
                    v = embed([text])
                    self._vectors = _replace_row_aligned(self._vectors, i, v)
                    self._persist()
                    return True
            return False

    
    def query(self, text: str, top_k: int = RAG_TOP_K) -> List[Dict]:
        with self._lock:
            if not self._docs or self._vectors is None:
                return []
            q = embed(text, is_query=True)  # (1, dq)
            mat = _align_dims(self._vectors, q.shape[1])
            q = _align_dims(q, mat.shape[1])
            sims = (mat @ q.T).ravel()
            top_k = min(top_k, len(self._docs))
            idx = np.argsort(-sims)[:top_k]
            return [
                {"text": self._docs[i]["text"],
                 "meta": self._docs[i].get("meta", {}),
                 "score": float(sims[i])}
                for i in idx
            ]

    def count(self) -> int:
        return len(self._docs)


def _align_dims(mat: np.ndarray, dim: int) -> np.ndarray:
    if mat.shape[1] == dim:
        return mat
    if mat.shape[1] < dim:
        pad = np.zeros((mat.shape[0], dim - mat.shape[1]), dtype=np.float32)
        return np.hstack([mat, pad])
    return mat[:, :dim]


def _vstack_aligned(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    dim = max(a.shape[1], b.shape[1])
    return np.vstack([_align_dims(a, dim), _align_dims(b, dim)])


def _replace_row_aligned(mat: np.ndarray, i: int, row: np.ndarray) -> np.ndarray:
    dim = max(mat.shape[1], row.shape[1])
    mat = _align_dims(mat, dim)
    mat[i] = _align_dims(row, dim)[0]
    return mat


_stores: Dict[str, LocalVectorStore] = {}
_stores_lock = threading.Lock()


def get_store(name: str) -> LocalVectorStore:
    with _stores_lock:
        if name not in _stores:
            _stores[name] = LocalVectorStore(name)
        return _stores[name]


if __name__ == "__main__":
    s = get_store("_smoketest")
    s.add(["AI 工程师的职业路径", "深度学习课程大纲", "报名费用与优惠"])
    for r in s.query("学习深度学习要多少钱"):
        print(f"{r['score']:.3f}  {r['text']}")
