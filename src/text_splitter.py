from typing import List
def split_text_into_chunks(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """
    Splits text into overlapping chunks.
    Args:
        text: Full extracted text.
        chunk_size: Size of each chunk in characters.
        overlap: Number of overlapping characters.

    Returns:
        List of text chunks.
    """
    if not text.strip():
        return []
    
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap # To create overlap between chunks
    return chunks