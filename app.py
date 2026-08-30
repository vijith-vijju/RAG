# app.py
# Streamlit UI for the Bilingual Telugu-English RAG Assistant

import streamlit as st
from rag_engine import (
    extract_text_from_pdf,
    split_into_chunks,
    build_vector_store,
    retrieve_chunks,
    detect_language,
    answer_query,
)

# -------------------------------------------------------------------
# Page config
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Bilingual RAG Assistant",
    page_icon="📄",
    layout="centered"
)

st.title("📄 Bilingual RAG Assistant")
st.markdown("Ask questions in **Telugu** or **English** from your uploaded PDFs.")
st.divider()

# -------------------------------------------------------------------
# Sidebar: API key input
# -------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    gemini_api_key = st.text_input(
        "Gemini API Key",
        type="password",
        placeholder="Paste your Gemini API key here"
    )
    st.markdown("[Get a free Gemini API key →](https://aistudio.google.com/app/apikey)")
    st.divider()
    st.caption("Your key is used only for this session and never stored.")

# -------------------------------------------------------------------
# Section 1: Upload PDFs
# -------------------------------------------------------------------
st.subheader("1️⃣ Upload PDFs")

uploaded_files = st.file_uploader(
    "Upload one or more PDFs (Telugu / English)",
    type=["pdf"],
    accept_multiple_files=True
)

# -------------------------------------------------------------------
# Section 2: Ingest documents (build vector store)
# -------------------------------------------------------------------
if uploaded_files:
    if st.button("📥 Ingest Documents", type="primary"):
        with st.spinner("Reading PDFs, chunking, and building vector store..."):
            all_chunks = []

            for pdf_file in uploaded_files:
                text = extract_text_from_pdf(pdf_file)

                if not text.strip():
                    st.warning(f"⚠️ Could not extract text from **{pdf_file.name}**. It may be a scanned image PDF.")
                    continue

                chunks = split_into_chunks(text, source_name=pdf_file.name)
                all_chunks.extend(chunks)

            if all_chunks:
                vector_store = build_vector_store(all_chunks)

                st.session_state["vector_store"] = vector_store
                st.session_state["num_chunks"] = len(all_chunks)

                st.success(f"✅ Ingested {len(uploaded_files)} PDF(s) → {len(all_chunks)} chunks stored in vector DB.")
            else:
                st.error("No text could be extracted from the uploaded PDFs.")

# -------------------------------------------------------------------
# Section 3: Ask a question
# -------------------------------------------------------------------
st.divider()
st.subheader("2️⃣ Ask a Question")

query = st.text_input(
    "Enter your question (Telugu or English)",
    placeholder="e.g., కర్మయోగం అంటే ఏమిటి? or What is Karma Yoga?"
)

if st.button("🔍 Get Answer", type="primary"):

    if not gemini_api_key:
        st.error("Please enter your Gemini API key in the sidebar.")
    elif "vector_store" not in st.session_state:
        st.error("Please upload and ingest PDFs first.")
    elif not query.strip():
        st.error("Please enter a question.")
    else:
        # detect and display query language before searching
        detected_lang = detect_language(query)
        st.info(f"🌐 Detected Language: **{detected_lang}**")

        with st.spinner("Searching documents and generating answer..."):
            # retrieve_chunks returns docs
            docs = retrieve_chunks(st.session_state["vector_store"], query, k=3)

            answer = answer_query(query, docs, gemini_api_key)


        # Display: Answer
        
        st.divider()
        st.subheader("💬 Answer")
        st.write(answer)

        # Display: Sources with similarity scores
        st.subheader("📚 Sources Used")
        st.caption("These are the exact chunks retrieved from your PDFs.")

        for i, doc in enumerate(docs, start=1):
            source   = doc.metadata.get("source", "Unknown")
            chunk_id = doc.metadata.get("chunk_id", "?")
            language = doc.metadata.get("language", "?")

            # show similarity score in the expander title
            with st.expander(f"Source {i} — {source}  |  Lang: {language}  |  Chunk #{chunk_id}"):
                st.write(doc.page_content)