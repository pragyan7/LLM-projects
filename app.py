import os
import tempfile
import streamlit as st

from src.pdf_utils import extract_text_from_pdf
from src.text_splitter import split_text_into_chunks
from src.embeddings import load_embedding_model, embed_text_chunks
from src.retriever import build_faiss_index, retrieve_top_k_chunks

st.set_page_config(page_title="Chat with Your PDFs", page_icon="📄")
st.title("Chat with Your PDFs 📄")
st.write("Upload a PDF and ask questions about its content!")

@st.cache_resource
def get_model():
    return load_embedding_model()

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_pdf_path = tmp_file.name

    try:
        with st.spinner("Extracting text from PDF..."):
            pdf_text = extract_text_from_pdf(temp_pdf_path)
        
        if not pdf_text.strip():
            st.error("no readable text found in the PDF.")
            st.stop()
        
        with st.spinner("Splitting text into chunks..."):
            chunks = split_text_into_chunks(pdf_text, chunk_size=500, overlap=100)
        
        if not chunks:
            st.error("No chunks were created from the PDF text.")
            st.stop()
        
        with st.spinner("Generating embeddings..."):
            model = get_model()
            embeddings = embed_text_chunks(model, chunks)
        
        with st.spinner("Building FAISS index..."):
            index = build_faiss_index(embeddings)
        
        st.success(f"PDF processed successfully! Total chunks created: {len(chunks)}")

        query = st.text_input("Ask a question about the PDF")

        if query:
            with st.spinner("Retrieving relevant chunks..."):
                results = retrieve_top_k_chunks(query, model, index, chunks, k=3)

            st.subheader("Top relevant chunks")
            for i, (chunk, score) in enumerate(results, start=1):
                st.markdown(f"**Chunk {i}** \nDistance: `{score:.4f}`")
                st.write(chunk)
                st.markdown("---")
    finally:
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)