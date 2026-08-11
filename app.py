import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from transformers import pipeline

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

        with st.spinner("Processing PDF..."):

            # Load PDF
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()

            # Split Text
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50
            )

            docs = splitter.split_documents(documents)

            # Embeddings
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

            # Vector Store
            db = FAISS.from_documents(
                docs,
                embeddings
            )

            # Similarity Search
            results = db.similarity_search(
                question,
                k=3
            )

            context = "\n".join(
                [doc.page_content for doc in results]
            )

            # Load LLM
            generator = pipeline(
                task="text2text-generation",
                model="google/flan-t5-base"
            )

            prompt = f"""
            Answer the question using only the context below.

            Context:
            {context}

            Question:
            {question}
            """

            answer = generator(
                prompt,
                max_length=200,
                do_sample=False
            )

            st.subheader("Answer")

            st.write(
                answer[0]["generated_text"]
            )

            st.subheader("Retrieved Chunks")

            for i, doc in enumerate(results):
                st.write(f"Chunk {i+1}")
                st.info(doc.page_content[:500])
