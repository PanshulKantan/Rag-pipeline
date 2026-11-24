# RAG Pipeline

A Python Retrieval-Augmented Generation (RAG) pipeline that answers questions using context retrieved from a research paper.

## Overview

The pipeline reads a pdf, extracts text from each page, splits the text into overlapping chunks, and stores embeddings in a FAISS in-memory vector database. When a question is entered, the script retrieves the three closest chunks and sends them with the question to the Gemini API.

The generated answer is instructed to:

- Greet Panshul Kantan as the research paper's author
- Answer only from the retrieved context
- Say `I don't know.` when the answer is unavailable

## Features

- PDF text extraction with PyPDF
- Recursive text chunking with LangChain
- Semantic embeddings using `all-MiniLM-L6-v2`
- Similarity search using a FAISS `IndexFlatL2`
- Context-grounded answer generation with Google Gen AI
- Interactive command-line questions

## Project Structure

```text
Rag pipeline/
|-- again.py
|-- pdf
|-- README.md
`-- .vscode/
```

## Requirements

- Python 3.10 or newer
- Internet access for downloading the embedding model and calling the Gemini API
- A Google Gemini API key

Python packages:

```text
pypdf
langchain-text-splitters
sentence-transformers
numpy
faiss-cpu
google-genai
```

## Installation

Clone the repository and enter the project directory:

```powershell
cd "Rag pipeline"
```

Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks the activation script, activate it with Command Prompt:

```bat
.venv\Scripts\activate.bat
```

Install the required packages:

```powershell
pip install pypdf langchain-text-splitters sentence-transformers numpy faiss-cpu google-genai
```

## API Configuration

Open `again.py` and replace the placeholder API key:

```python
gemini_client = genai.Client(api_key="YOUR_API_KEY")
```

Do not commit a real API key to the repository. If this is a public repository, restrict the key and revoke it if it has already been exposed.

## Usage

Run the script from the project directory:

```powershell
python again.py
```

Enter a question when prompted:

```text
Your question: What is the objective of this research?
```

The script prints the embedding shape, number of vectors in the FAISS database, the indices of the retrieved chunks, and the generated answer.

## How It Works

1. PyPDF extracts text from every page of the pdf.
2. `RecursiveCharacterTextSplitter` divides the text into 1,000-character chunks with a 200-character overlap.
3. `SentenceTransformer` creates 384-dimensional embeddings with `all-MiniLM-L6-v2`.
4. FAISS stores the embeddings in an in-memory `IndexFlatL2` database.
5. The question is embedded and matched against the stored chunks.
6. The three nearest chunks are combined into a prompt.
7. Google Gemini generates an answer from that prompt.
