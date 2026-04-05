import fitz # PyMuPDF library for PDF processing

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract all text from a PDF file.
    Args:
        pdf_path: The path to the PDF file from which to extract text.
    Returns: Extracted text as a single string.
    """
    text = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            page_text = page.get_text("text")
            if page_text:
                text.append(page_text)
    return "\n".join(text)