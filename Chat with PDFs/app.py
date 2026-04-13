import os
import warnings
import tempfile
import streamlit as st
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=DeprecationWarning)

load_dotenv()
print("DEBUG KEY EXISTS:", bool(os.getenv("OPENROUTER_API_KEY")))
print("DEBUG KEY PREFIX:", os.getenv("OPENROUTER_API_KEY", "")[:12])

from src.pdf_utils import extract_text_from_pdf
from src.text_splitter import split_text_into_chunks
from src.embeddings import load_embedding_model 
from src.retriever import build_faiss_index, retrieve_top_k_chunks
from src.llm_client import ask_openrouter

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
            pages = extract_text_from_pdf(temp_pdf_path)
        
        if not pages:
            st.error("no readable text found in the PDF.")
            st.stop()
        
        with st.spinner("Splitting text into chunks..."):
            chunks = split_text_into_chunks(pages)
        
        if not chunks:
            st.error("No chunks were created from the PDF text.")
            st.stop()
        
        with st.spinner("Generating embeddings..."):
            model = get_model()
            texts = [c['text'] for c in chunks]
            embeddings = model.encode(texts)
        
        with st.spinner("Building FAISS index..."):
            index = build_faiss_index(embeddings)
        
        st.success(f"PDF processed successfully! Total chunks created: {len(chunks)}")

        query = st.text_input("Ask a question about the PDF")

        def generate_answer(query, retrieved_chunks):
                context = "\n\n".join([chunk['text'] for chunk, _ in retrieved_chunks])

                answer = f"""
                **Question**: {query}
                Answer based on the document:
                **Context**: {context}
                """
                return answer
        
        if query:
            with st.spinner("Retrieving relevant chunks..."):
                results = retrieve_top_k_chunks(query, model, index, chunks, k=3)
            
            with st.spinner("Generating answer with OpenRouter..."):
                answer = ask_openrouter(
                    query, 
                    results, 
                    model="nvidia/nemotron-3-super-120b-a12b:free"
                    )
        
            st.subheader("Answer")
            st.write(answer)
            
            with st.expander("Sources used"):
                for i, (chunk, score) in enumerate(results, start=1):
                    st.markdown(
                        f"**Chunk {i}** - Page {chunk['page']} - distance `{score:.4f}`"
                        )
                    st.write(chunk['text'])
                    st.markdown("---")
    finally:
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)