import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

class SHLCatalog:
    def __init__(self, json_path="shl_assessments.json"):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.assessments = self._load(json_path)
        self.index = self._build_index()

    def _load(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError("Run scraper first to generate shl_assessments.json")
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _build_index(self):
        if not self.assessments:
            return None
        texts = [f"{a['name']} {a.get('description', '')}" for a in self.assessments]
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings.astype(np.float32))
        return index, embeddings

    def search(self, query: str, top_k: int = 15):
        if not self.index:
            return []
        query_vec = self.model.encode([query], normalize_embeddings=True)
        distances, indices = self.index[0].search(query_vec.astype(np.float32), top_k)
        results = [self.assessments[i] for i in indices[0] if i < len(self.assessments)]
        return results
