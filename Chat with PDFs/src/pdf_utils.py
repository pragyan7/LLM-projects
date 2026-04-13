import fitz # PyMuPDF library for PDF processing

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text page-wise from a PDF file.
    Args:
        pdf_path: The path to the PDF file from which to extract text.
    Returns: A list of tuples containing the extracted text and page number for each page.
    """
    pages = []
    
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text = page.get_text("text")
            if text.strip():  # Only consider pages with non-empty text
                pages.append((text, page.number))

    return pages