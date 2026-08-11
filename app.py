import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="AI PDF Assistant",
    page_icon="🤖",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

.stApp {
    background-color: #f4f7fc;
}

/* Header */
.main-title {
    text-align: center;
    font-size: 50px;
    font-weight: 800;
    color: #2563eb;
}

.sub-title {
    text-align: center;
    font-size: 18px;
    color: #475569;
    margin-bottom: 30px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #1e3a8a;
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

/* Cards */
.card {
    background: white;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    border: 1px solid #e2e8f0;
    box-shadow: 0px 3px 10px rgba(0,0,0,0.08);
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: white;
    padding: 15px;
    border: 2px solid #bfdbfe;
    border-radius: 15px;
}

/* Text box */
.stTextInput input {
    background: white !important;
    color: black !important;
    border: 2px solid #60a5fa !important;
    border-radius: 12px !important;
}

/* Button */
.stButton button {
    background: #2563eb !important;
    color: white !important;
    border-radius: 10px;
    height: 50px;
    font-weight: bold;
    width: 100%;
}

/* Answer box */
.answer-box {
    background: white;
    border-left: 6px solid #2563eb;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0px 3px 10px rgba(0,0,0,0.08);
    color: black;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.title("🤖 AI PDF Assistant")

    st.markdown("""
### Features

✅ Upload PDF

✅ Semantic Search

✅ NLP Processing

✅ FAISS Vector Search

✅ LangChain Integration

✅ Streamlit Deployment

---

### Workflow

📄 PDF Upload

⬇️

✂️ Text Chunking

⬇️

🧠 Embeddings

⬇️

📚 FAISS Search

⬇️

🎯 Answer Retrieval
""")

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown("""
<div class="main-title">
🤖 AI PDF Assistant
</div>

<div class="sub-title">
Chat with your PDF using NLP, FAISS and LangChain
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------

st.markdown("## 📊 System Overview")

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class='card'>
    <h3>🧠 Model</h3>
    <h2>MiniLM</h2>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class='card'>
    <h3>📚 Database</h3>
    <h2>FAISS</h2>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class='card'>
    <h3>⚡ Framework</h3>
    <h2>LangChain</h2>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --------------------------------------------------
# INPUT SECTION
# --------------------------------------------------

left, right = st.columns([1,1])

with left:
    uploaded_file = st.file_uploader(
        "📄 Upload PDF",
        type=["pdf"]
    )

with right:
    question = st.text_input(
        "❓ Ask a Question"
    )

# --------------------------------------------------
# PROCESS
# --------------------------------------------------

if uploaded_file:

    pdf_path = "temp.pdf"

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("✅ PDF Uploaded Successfully")

    if st.button("🚀 Generate Answer"):

        if not question.strip():
            st.warning("Enter a question first.")
            st.stop()

        with st.spinner("Processing PDF..."):

            loader = PyPDFLoader(pdf_path)
            documents = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50
            )

            docs = splitter.split_documents(documents)

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

                st.markdown("## 📖 Retrieved Context")

                for i, doc in enumerate(results):

                    with st.expander(
                        f"Context Chunk {i+1}"
                    ):
                        st.write(doc.page_content)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("---")

st.markdown("""
### 👨‍💻 Data Science Portfolio Project

Built with:

- Streamlit
- LangChain
- FAISS
- NLP
- RAG Architecture
- HuggingFace Embeddings
""")
