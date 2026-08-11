import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# ---------------------------------
# PAGE CONFIG
# ---------------------------------

st.set_page_config(
    page_title="AI PDF RAG Assistant",
    page_icon="🤖",
    layout="wide"
)

# ---------------------------------
# CUSTOM CSS
# ---------------------------------

st.markdown("""
<style>

/* Main Background */
.stApp{
    background-color:#0E1117;
    color:white;
}

/* Title */
.main-title{
    text-align:center;
    font-size:3rem;
    font-weight:700;
    color:#4F8BF9;
    margin-top:10px;
}

/* Subtitle */
.sub-title{
    text-align:center;
    color:#B0B0B0;
    margin-bottom:30px;
}

/* Answer Box */
.answer-box{
    background:#1E293B;
    padding:20px;
    border-radius:15px;
    border-left:5px solid #4F8BF9;
    color:white;
    font-size:16px;
}

/* Card */
.card{
    background:#111827;
    padding:15px;
    border-radius:12px;
    border:1px solid #1F2937;
}

/* Sidebar */
section[data-testid="stSidebar"]{
    background-color:#111827;
}

/* Buttons */
.stButton button{
    background-color:#2563EB;
    color:white;
    border:none;
    border-radius:10px;
    height:50px;
    width:100%;
    font-weight:bold;
}

/* Input */
.stTextInput div div input{
    background-color:#1F2937;
    color:white;
}

/* Upload */
[data-testid="stFileUploader"]{
    background:#111827;
    padding:15px;
    border-radius:10px;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------
# SIDEBAR
# ---------------------------------

with st.sidebar:

    st.title("📚 Project Info")

    st.markdown("""
### AI PDF RAG Assistant

This project demonstrates:

✅ NLP

✅ Embeddings

✅ Semantic Search

✅ Vector Database (FAISS)

✅ Retrieval-Augmented Generation (RAG)

✅ LangChain Framework

✅ Streamlit Deployment
""")

    st.divider()

    st.markdown("""
### Instructions

1. Upload a PDF

2. Ask a question

3. Click Generate Answer

4. View the retrieved content

### Tech Stack

- Python
- Streamlit
- LangChain
- FAISS
- Sentence Transformers
""")

# ---------------------------------
# HEADER
# ---------------------------------

st.markdown(
    '<div class="main-title">🤖 AI PDF RAG Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">Ask intelligent questions from your documents using NLP and Vector Search</div>',
    unsafe_allow_html=True
)

# ---------------------------------
# METRICS
# ---------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Embedding Model", "MiniLM")

with col2:
    st.metric("Vector DB", "FAISS")

with col3:
    st.metric("Framework", "LangChain")

st.markdown("---")

# ---------------------------------
# PDF UPLOAD
# ---------------------------------

uploaded_file = st.file_uploader(
    "📄 Upload PDF",
    type=["pdf"]
)

question = st.text_input(
    "❓ Ask a Question"
)

if uploaded_file:

    pdf_path = "temp.pdf"

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("PDF Uploaded Successfully ✅")

    if st.button("🚀 Generate Answer"):

        if not question.strip():
            st.warning("Please enter a question.")
            st.stop()

        with st.spinner("Analyzing Document..."):

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

            # Vector Database
            db = FAISS.from_documents(
                docs,
                embeddings
            )

            # Search
            results = db.similarity_search(
                question,
                k=3
            )

            # Answer
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
                st.warning("No answer found.")

            st.markdown("---")

            # Context
            st.markdown("## 📖 Retrieved Context")

            for i, doc in enumerate(results):

                with st.expander(f"Chunk {i+1}"):

                    st.write(doc.page_content)
