# AskMyDocs

AskMyDocs is a local AI document assistant. It lets you ask questions across multiple PDFs or generate a cited summary of one PDF.

## Features

- Upload and search multiple text-based PDFs.
- Ask questions and see the PDF passages used to form an answer.
- Generate a summary of a selected PDF.
- Show document, page, and searchable passage counts.
- Run the language model, embeddings, and vector search locally.

## How it works

1. `pypdf` extracts text and page numbers from each PDF.
2. The app splits the text into overlapping passages.
3. Ollama's `nomic-embed-text` model creates embeddings for those passages.
4. ChromaDB finds passages related to a question.
5. Ollama's `llama3.2:3b` model answers using the retrieved passages.

## Requirements

- Python 3.14
- [Ollama](https://ollama.com/download)
- The Ollama models `llama3.2:3b` and `nomic-embed-text`

No paid AI API key is required.

## Run on Windows

Open PowerShell in the project folder and run:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
ollama pull llama3.2:3b
ollama pull nomic-embed-text
.\.venv\Scripts\python.exe -m streamlit run app.py