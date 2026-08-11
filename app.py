import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="AI PDF RAG Assistant",
    page_icon="🤖",
    layout="wide"
)

# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown("""
<style>

/* Main App */
.stApp {
    background: linear-gradient(
        135deg,
        #0f172a,
        #111827,
        #1e293b
    );
    color: white;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #111827;
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

/* Main Title */
.main-title {
    text-align: center;
    font-size: 3.2rem;
    font-weight: bold;
    color: #60A5FA;
    margin-bottom: 10px;
}

/* Subtitle */
.sub-title {
    text-align: center;
    color: #E5E7EB;
    font-size: 1.2rem;
    margin-bottom: 35px;
}

/* Upload Container */
[data-testid="stFileUploader"] {
    background-color: #1F2937;
    padding: 20px;
    border-radius: 15px;
}

/* Input Box */
.stTextInput input {
    background-color: #1F2937 !important;
    color: white !important;
    border: 1px solid #3B82F6 !important;
    border-radius: 10px !important;
}

/* Buttons */
.stButton button {
    width: 100%;
    height: 55px;
    border-radius: 12px;
    background: linear-gradient(
        to right,
        #2563EB,
        #3B82F6
    );
    color: white;
    font-weight: bold;
    font-size: 16px;
    border: none;
}

/* Answer Box */
.answer-box {
    background-color: #1E293B;
    border-left: 6px solid #60A5FA;
    padding: 20px;
    border-radius: 12px;
    color: white;
    font-size: 16px;
    line-height: 1.6;
}

/* Metrics */
[data-testid="metric-container"] {
    background-color: #1F2937;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #374151;
}

/* Expander */
.streamlit-expanderHeader {
    color: white !important;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.title("📚 Project Info")

    st.markdown("""
### AI PDF RAG Assistant

A Retrieval-Augmented Generation (RAG) project built using:

✅ NLP

✅ Sentence Transformers

✅ Semantic Search

✅ FAISS Vector Database

✅ LangChain

✅ Streamlit
""")

    st.divider()

    st.markdown("""
### How To Use

1️⃣ Upload a PDF

2️⃣ Ask a question

3️⃣ Click Generate Answer

4️⃣ View retrieved context

---

### Technologies

- Python
- LangChain
- FAISS
- Streamlit
- HuggingFace Embeddings
""")

# ==========================================================
# HEADER
# ==========================================================

st.markdown(
"""
<div class="main-title">
🤖 AI PDF RAG Assistant
</div>

<div class="sub-title">
Ask intelligent questions from your PDF documents using NLP, Embeddings and Vector Search
</div>
""",
unsafe_allow_html=True
)

# ==========================================================
# DASHBOARD METRICS
# ==========================================================

st.markdown("## 📊 System Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Embedding Model",
        "MiniLM"
    )

with col2:
    st.metric(
        "Vector Database",
        "FAISS"
    )

with col3:
    st.metric(
        "Framework",
        "LangChain"
    )

st.markdown("---")

# ==========================================================
# FILE UPLOAD
# ==========================================================

st.markdown("## 📄 Upload Your PDF")

uploaded_file = st.file_uploader(
    "",
    type=["pdf"]
)

# ==========================================================
# QUESTION INPUT
# ==========================================================

question = st.text_input(
    "❓ Ask a Question"
)

# ==========================================================
# PROCESS
# ==========================================================

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

        with st.spinner("🔍 Searching PDF Content..."):

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
                    f"Chunk {i+1}"
                ):

                    st.write(
                        doc.page_content
                    )

# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.caption(
    "🚀 Built using Streamlit • LangChain • FAISS • Sentence Transformers"
)
