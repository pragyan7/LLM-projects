from typing import List

def split_text_into_chunks(pages, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """
    Chunks are created page-wise to maintain context, and each chunk includes an overlap with the previous chunk to preserve continuity.
    Args:
        text: Full extracted text.
        chunk_size: Size of each chunk in characters.
        overlap: Number of overlapping characters.

    Returns:
        List of dicts:
        {
            "text": "...",
            "page": int
        }
    """
    chunks = []
    
    for page_text, page_num in pages:
        start = 0

        while start < len(page_text):
            end = start + chunk_size
            chunk_text = page_text[start:end].strip()

            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "page": page_num + 1 # make it human-readable
                })
            start += chunk_size - overlap # To create overlap between chunks
    return chunks