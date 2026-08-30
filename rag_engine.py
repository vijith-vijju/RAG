from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langdetect import detect
from google import genai


# STEP 1: Extract text from a PDF file

def extract_text_from_pdf(pdf_file) -> str:
    """
    Reads a PDF (uploaded via Streamlit) and returns all its text as one string.
    pdf_file: a file-like object from st.file_uploader
    """
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:  # some pages may be empty / image-only
            text += page_text + "\n"
    return text


# STEP 2: Split text into smaller overlapping chunks

def split_into_chunks(text: str, source_name: str) -> list[dict]:
    """
    Splits a large text into chunks of ~500 characters with 100-char overlap.

    Each chunk stores:
      - source:    which PDF it came from
      - chunk_id:  its position in the list (e.g. chunk 0, 1, 2 ...)
      - language:  auto-detected language of that chunk ("te" = Telugu, "en" = English)
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    raw_chunks = splitter.split_text(text)

    chunks = []
    for idx, chunk in enumerate(raw_chunks):
        try:
            lang = detect(chunk)   # returns "te", "en", "hi", etc.
        except Exception:
            lang = "unknown"       # fallback if detection fails (e.g. very short chunk)

        chunks.append({
            "text":     chunk,
            "source":   source_name,
            "chunk_id": idx,
            "language": lang,
        })

    return chunks


# STEP 3: Build a FAISS vector store from chunks

def build_vector_store(all_chunks: list[dict]):
    """
    Converts all text chunks into embeddings and stores them in FAISS.

    Model used: paraphrase-multilingual-MiniLM-L12-v2
    - Free, runs locally (no API key needed)
    - Supports both Telugu and English
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    texts = [chunk["text"] for chunk in all_chunks]

    metadatas = [
        {
            "source":   chunk["source"],
            "chunk_id": chunk["chunk_id"],
            "language": chunk["language"],
        }
        for chunk in all_chunks
    ]

    vector_store = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    return vector_store


# STEP 4: Retrieve relevant chunks WITH similarity scores

def retrieve_chunks(vector_store, query: str, k: int = 3):
    """
    Returns a list of (Document, score) tuples.
    Lower score = more similar in FAISS (it returns L2 distance by default).
    We convert to a 0–1 similarity value for display: similarity = 1 / (1 + score)
    """
    results = vector_store.similarity_search_with_score(query, k=k)

    # Attach a human-readable similarity score to each document's metadata
    docs_with_scores = []
    for doc, raw_score in results:
        # Convert L2 distance → similarity (higher is better, max ~1.0)
        similarity = round((1 / (1 + raw_score))*100, 2)
        doc.metadata["similarity"] = similarity
        docs_with_scores.append(doc)

    return docs_with_scores


# STEP 5: Detect language of the user's query

def detect_language(text: str) -> str:
    # Returns a friendly display string.
    try:
        code = detect(text)
        # Map language codes to readable names
        lang_map = {
            "te": "Telugu 🇮🇳",
            "en": "English 🇬🇧",
            "hi": "Hindi 🇮🇳",
        }
        return lang_map.get(code, f"Other ({code})")
    except Exception:
        return "Unknown"


# STEP 6: Answer the query using Gemini, grounded in retrieved chunks

def answer_query(query: str, docs: list, gemini_api_key: str) -> str:
    # Build context, labeling each chunk with its source filename
    context_parts = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata["source"]
        context_parts.append(f"[Source {i} — {source}]\n{doc.page_content}")
    context = "\n\n".join(context_parts)

    prompt = f"""You are a helpful assistant. Answer the question using ONLY the context provided below.
When answering, mention which source supports the answer (e.g., "According to <filename>, ...").
If the answer is not found in the context, say: "This information is not available in the uploaded documents."
Do not make up any information.

Context:
{context}

Question: {query}

Answer:"""

    client = genai.Client(api_key=gemini_api_key)
    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt
    )

    return response.text