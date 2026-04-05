from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"

def load_embedding_model() -> SentenceTransformer:
    """
    Load and return a SentenceTransformer model for generating embeddings.
    """
    return SentenceTransformer(MODEL_NAME)

def embed_text_chunks(model: SentenceTransformer, chunks: list[str]):
    """
    Generate embeddings for a list of text chunks using the provided model.
    """
    return model.encode(chunks, convert_to_numpy=True)