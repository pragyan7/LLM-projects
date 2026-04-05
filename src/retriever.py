import faiss
import numpy as np
from typing import List, Tuple

def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    """
    Build a FAISS index from the given embeddings.
    
    Args:
        embeddings: Numpy array of sape (n_chunks, embedding_dim)

    Returns:
        FAISS index containing the embeddings.
    """
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype("float32"))
    return index

def retrieve_top_k_chunks(
        query: str,
        model,
        index,
        chunks: List[str],
        k: int = 3
) -> List[Tuple[str, float]]:
    """
    Retrieve top-k most relevant chunks for a query.

    Args:
        query: User query
        model: FAISS index
        index: FAISS index
        chunks: Original text chunks
        k: Number of chunks to retrieve
    
    Returns:
        List of (chunk_text, distance)
    """
    query_embedding = model.encode([query], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(query_embedding, k)

    results = []
    for idx, dist in zip(indices[0], distances[0]):
        if 0 <= idx < len(chunks):
            results.append((chunks[idx], float(dist)))
    return results