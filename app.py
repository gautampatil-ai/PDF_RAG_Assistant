import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="AI PDF Assistant",
    page_icon="🤖",
    layout="wide"
)

# =====================================================
# CSS
# =====================================================

st.markdown("""
<style>

/* Main background */
.stApp{
    background-color:#F5F9FF;
}

/* Sidebar */
section[data-testid="stSidebar"]{
    background:linear-gradient(180deg,#2563EB,#1D4ED8);
}

section[data-testid="stSidebar"] *{
    color:white !important;
}

/* Header */
.main-title{
    text-align:center;
    color:#1E40AF;
    font-size:55px;
    font-weight:bold;
}

.sub-title{
    text-align:center;
    color:#475569;
    font-size:18px;
    margin-bottom:25px;
}

/* Cards */
.metric-box{
    background:white;
    padding:20px;
    border-radius:15px;
    text-align:center;
    border:1px solid #E2E8F0;
    box-shadow:0px 4px 10px rgba(0,0,0,0.08);
}

/* Upload Area */
[data-testid="stFileUploader"]{
    background:white;
    padding:15px;
    border-radius:15px;
    border:2px solid #BFDBFE;
}

/* Input */
.stTextInput input{
    background:white !important;
    color:black !important;
    border:2px solid #60A5FA !important;
    border-radius:10px !important;
}

/* Button */
.stButton button{
    background:#2563EB !important;
    color:white !important;
    width:100%;
    height:50px;
    border:none;
    border-radius:10px;
    font-size:18px;
    font-weight:bold;
}

/* Answer Box */
.answer-box{
    background:white;
    padding:20px;
    border-radius:15px;
    border-left:6px solid #2563EB;
    box-shadow:0px 4px 10px rgba(0,0,0,0.08);
    color:black;
    font-size:16px;
}

/* Headings */
h1,h2,h3{
    color:#1E293B !important;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.title("🤖 AI PDF Assistant")

    st.markdown("""
### About

This RAG project uses:

✅ NLP

✅ LangChain

✅ FAISS

✅ HuggingFace Embeddings

✅ Semantic Search

✅ Streamlit

---

### How To Use

1. Upload PDF

2. Enter Question

3. Click Generate Answer

4. View Results

---

### Project Type

Data Science

Natural Language Processing

Retrieval Augmented Generation
""")

# =====================================================
# HEADER
# =====================================================

st.markdown("""
<div class="main-title">
🤖 AI PDF Assistant
</div>

<div class="sub-title">
Chat with your PDF using NLP, Embeddings and Vector Search
</div>
""", unsafe_allow_html=True)

# =====================================================
# METRICS
# =====================================================

st.markdown("## 📊 System Overview")

col1,col2,col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="metric-box">
        <h3>🧠 Model</h3>
        <h2>MiniLM</h2>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-box">
        <h3>📚 Database</h3>
        <h2>FAISS</h2>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-box">
        <h3>⚡ Framework</h3>
        <h2>LangChain</h2>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# =====================================================
# PDF + QUESTION
# =====================================================

left,right = st.columns(2)

with left:
    uploaded_file = st.file_uploader(
        "📄 Upload PDF",
        type=["pdf"]
    )

with right:
    question = st.text_input(
        "❓ Ask Question"
    )

# =====================================================
# PROCESS
# =====================================================

if uploaded_file:

    pdf_path="temp.pdf"

    with open(pdf_path,"wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("✅ PDF Uploaded Successfully")

    if st.button("🚀 Generate Answer"):

        if not question.strip():
            st.warning("Please enter a question.")
            st.stop()

        with st.spinner("Analyzing PDF..."):

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

            st.markdown("## 📖 Retrieved Context")

            for i, doc in enumerate(results):

                with st.expander(
                    f"Context Chunk {i+1}"
                ):

                    st.write(
                        doc.page_content
                    )

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.markdown("""
### 👨‍💻 Portfolio Project

Built using:

- Streamlit
- LangChain
- FAISS
- NLP
- RAG Architecture
- HuggingFace Embeddings

⭐ Ideal for Data Science & AI Portfolio
""")
