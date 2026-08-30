# Bilingual Telugu-English RAG Assistant

A Retrieval-Augmented Generation (RAG) system that enables question answering over Telugu and English PDF documents.

The application supports cross-lingual retrieval, allowing users to ask questions in Telugu or English and receive grounded answers based on uploaded documents.

## Features

* PDF document ingestion
* Automatic text extraction
* Semantic chunking
* Multilingual embeddings
* FAISS vector database
* Cross-lingual retrieval
* Telugu and English query support
* Gemini-powered answer generation
* Source-grounded responses with citations
* Streamlit web interface

## Tech Stack

* Python
* Streamlit
* LangChain
* FAISS
* Sentence Transformers
* Google Gemini API
* LangDetect

## Architecture

PDF Upload
→ Text Extraction
→ Chunking
→ Multilingual Embeddings
→ FAISS Vector Store
→ Semantic Retrieval
→ Gemini Generation
→ Source-Grounded Answer

## Example Use Cases

### English PDF → Telugu Question

Question:

ప్రజలు CBDC ను ఎందుకు ఉపయోగించాలి?

Answer:

The system retrieves relevant English document chunks and generates a grounded answer in Telugu.

### Telugu PDF → English Question

Question:

What is Karma Yoga?

Answer:

The system retrieves relevant Telugu content and generates an English response.

## Installation

```bash
pip install -r requirements.txt
streamlit run app.py
```
## Screenshorts

<img width="1918" height="1140" alt="Screenshot 2026-06-16 124704" src="https://github.com/user-attachments/assets/3321a53f-d92a-4a79-90db-8c0fb654f4d9" />
<img width="845" height="907" alt="Screenshot 2026-06-15 233951" src="https://github.com/user-attachments/assets/f741627e-4e05-4702-b634-fbd7f83084d3" />


## Future Improvements

* OCR support for scanned PDFs
* Metadata-aware retrieval
* Hybrid search (BM25 + Vector Search)
* Retrieval evaluation metrics
* Reranking pipelines

## Author

Venugopal
B.Tech, IIT Madras
