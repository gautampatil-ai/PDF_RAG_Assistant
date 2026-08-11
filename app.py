import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📚"
)

st.title("📚 PDF RAG Chatbot")

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type="pdf"
)

question = st.text_input(
    "Ask a question from the PDF"
)

if uploaded_file:

    pdf_path = "temp.pdf"

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("PDF Uploaded Successfully ✅")

    if st.button("Generate Answer"):

        if not question:
            st.warning("Please enter a question")
            st.stop()

        with st.spinner("Processing PDF..."):

            # Load PDF
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()

            # Split PDF into chunks
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50
            )

            docs = splitter.split_documents(documents)

            # Create embeddings
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

            # Create vector database
            db = FAISS.from_documents(
                docs,
                embeddings
            )

            # Search
            results = db.similarity_search(
                question,
                k=3
            )

            st.subheader("Answer")

            if results:
                st.write(results[0].page_content)
            else:
                st.write("No relevant answer found.")

            st.subheader("Retrieved Chunks")

            for i, doc in enumerate(results):
                st.markdown(f"### Chunk {i+1}")
                st.info(doc.page_content)
