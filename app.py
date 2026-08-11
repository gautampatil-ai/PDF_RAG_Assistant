import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# ===================================================
# PAGE CONFIG
# ===================================================

st.set_page_config(
    page_title="AI PDF RAG Assistant",
    page_icon="🤖",
    layout="wide"
)

# ===================================================
# CUSTOM CSS
# ===================================================

st.markdown("""
<style>

/* Main Background */
.stApp {
    background-color: #F8FAFC;
}

/* Header */
.main-title {
    font-size: 48px;
    font-weight: 700;
    color: #1E40AF;
    text-align: center;
}

.sub-title {
    text-align: center;
    font-size: 18px;
    color: #64748B;
    margin-bottom: 30px;
}

/* Cards */
.card {
    background-color: white;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #E2E8F0;
    box-shadow: 0px 2px 10px rgba(0,0,0,0.05);
    text-align: center;
}

/* Answer */
.answer-box {
    background: #EFF6FF;
    border-left: 6px solid #2563EB;
    padding: 20px;
    border-radius: 12px;
    font-size: 16px;
    color: #111827;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: white;
}

/* Button */
.stButton button {
    width: 100%;
    height: 50px;
    border-radius: 10px;
    background-color: #2563EB;
    color: white;
    font-weight: bold;
    border: none;
}

/* Input */
.stTextInput input {
    border-radius: 10px !important;
}

/* Upload */
[data-testid="stFileUploader"] {
    background-color: white;
    border-radius: 12px;
    padding: 15px;
    border: 1px solid #CBD5E1;
}

</style>
""", unsafe_allow_html=True)

# ===================================================
# SIDEBAR
# ===================================================

with st.sidebar:

    st.title("📚 Project Dashboard")

    st.markdown("""
### About Project

AI PDF RAG Assistant allows users to:

✅ Upload PDF files

✅ Search documents

✅ Retrieve relevant information

✅ Use semantic search

✅ Explore NLP applications

---

### Tech Stack

- Python
- Streamlit
- LangChain
- FAISS
- HuggingFace
- NLP

---

### Workflow

PDF Upload

⬇️

Text Chunking

⬇️

Embeddings

⬇️

FAISS Search

⬇️

Answer Retrieval
""")

# ===================================================
# HEADER
# ===================================================

st.markdown("""
<div class="main-title">
🤖 AI PDF RAG Assistant
</div>

<div class="sub-title">
Retrieval-Augmented Generation using NLP, Embeddings, FAISS and LangChain
</div>
""", unsafe_allow_html=True)

# ===================================================
# SYSTEM OVERVIEW
# ===================================================

st.markdown("## 📊 System Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.info("""
### 🧠 Embedding Model

MiniLM
""")

with col2:
    st.success("""
### 📚 Vector Database

FAISS
""")

with col3:
    st.warning("""
### ⚡ Framework

LangChain
""")

st.markdown("---")

# ===================================================
# INPUT SECTION
# ===================================================

left, right = st.columns(2)

with left:
    uploaded_file = st.file_uploader(
        "📄 Upload PDF",
        type=["pdf"]
    )

with right:
    question = st.text_input(
        "❓ Ask a Question"
    )

# ===================================================
# PROCESSING
# ===================================================

if uploaded_file:

    pdf_path = "temp.pdf"

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("✅ PDF Uploaded Successfully")

    if st.button("🚀 Generate Answer"):

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

            st.stop()

        with st.spinner("Processing PDF..."):

            loader = PyPDFLoader(pdf_path)
            documents = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50
            )

            docs = splitter.split_documents(
                documents
            )

            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

            db = FAISS.from_documents(
                docs,
                embeddings
            )

            results = db.similarity_search(
                question,
                k=3
            )

            st.markdown("## 🎯 Answer")

            if results:

                st.markdown(
                    f"""
                    <div class="answer-box">
                    {results[0].page_content}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.warning(
                    "No answer found."
                )

            st.markdown("---")

            st.markdown(
                "## 📖 Retrieved Context"
            )

            for i, doc in enumerate(results):

                with st.expander(
                    f"📄 Context Chunk {i+1}"
                ):

                    st.write(
