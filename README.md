# 🧠 Intelligent MCQ Generator from PDF Using RAG

A production-ready Python application that automatically generates high-quality, conceptual Multiple-Choice Questions (MCQs) from uploaded PDF textbooks or unit notes using **Retrieval-Augmented Generation (RAG)**, **LangChain**, **FAISS Vector Database**, **Google Gemini / OpenAI**, and **Streamlit**.

---

## 🌟 Key Features

- 📄 **PDF Processing & Semantic Chunking**: Extracts text from multi-page PDFs and chunks content intelligently using `RecursiveCharacterTextSplitter` (`chunk_size=1000`, `chunk_overlap=150`).
- ⚡ **Local Vector Search with FAISS**: Embeds text chunks and builds an in-memory vector index to retrieve key concepts for question generation.
- 🎯 **RAG MCQ Generation**: Enforces strictly structured JSON output containing conceptual questions, 4 distinct choices (A, B, C, D), correct answer key, detailed reasoning explanations, and source reference text.
- 🤖 **Multi-LLM Support**: Supports **Google Gemini API** (`gemini-1.5-flash`) as primary, with optional support for **OpenAI API** (`gpt-4o-mini`).
- 📝 **Interactive Quiz & Self-Assessment UI**:
  - Customizable question count (3, 5, 10, 15, 20) and difficulty levels (Easy, Medium, Hard).
  - Interactive radio buttons for taking the quiz.
  - Automatic scoring with color-coded feedback (Green for Correct, Red for Incorrect).
  - Expandable **Source Reference Accordion** for full RAG traceability.
- 📥 **Export Capabilities**: Export generated quizzes and answer keys to JSON and TXT formats.
- 🔒 **State Management**: Built using `st.session_state` so vector indices and quiz responses persist seamlessly across UI interactions.

---

## 🏗️ Architecture & Project Structure

```
c:\Users\SUMAN\Downloads\mcq\
├── app.py                  # Main Streamlit UI layout and event loops
├── utils/
│   ├── __init__.py         # Package initializer
│   ├── pdf_processor.py    # PDF extraction & RecursiveCharacterTextSplitter chunking
│   ├── vector_store.py     # FAISS vector database setup & MMR retriever
│   └── mcq_generator.py   # LangChain chain, prompt engineering & structured JSON parsing
├── requirements.txt        # Exact dependencies
├── .env.example            # Environment variables template
├── .env                    # Local environment variables with your API key
└── README.md               # Documentation & Setup guide
```

---

## 🚀 Quickstart & Setup Guide

### 1. Prerequisites
- **Python 3.10+** installed on your system.
- An API Key for **Google Gemini API** (Get one at [Google AI Studio](https://aistudio.google.com/)) or **OpenAI API**.

### 2. Clone / Setup Project Directory
Navigate to your project directory:
```bash
cd c:\Users\SUMAN\Downloads\mcq
```

### 3. Create & Activate a Virtual Environment
```bash
# Windows PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Open `.env` and paste your Google Gemini or OpenAI API Key:
```env
GOOGLE_API_KEY=your_actual_google_api_key_here
```
*(Note: You can also enter your API key directly in the Streamlit sidebar at runtime).*

---

## 🏃 Running the Application

Launch the Streamlit app:
```bash
python -m streamlit run app.py
```


Open your browser at `http://localhost:8501`.

---

## 📖 Usage Instructions

1. **Configure Settings**:
   - In the sidebar, select your LLM Provider (Google Gemini recommended).
   - Enter your API Key (if not loaded from `.env`).
   - Select the desired **Number of Questions** and **Difficulty Level**.
2. **Upload PDF**:
   - Upload your unit or chapter PDF file using the uploader in the sidebar.
   - Click **"🚀 Process PDF & Index RAG"**.
3. **Generate Quiz**:
   - Click **"✨ Generate MCQs Now"** to execute the RAG retrieval pipeline and generate questions.
4. **Take Assessment**:
   - Select your answers for each question and click **"✅ Submit Quiz & View Score"**.
   - Review your total score, detailed explanations, and source context excerpts.
5. **Export**:
   - Download the quiz in **JSON** or **TXT** format using the download buttons.

---

## 🛡️ Edge Case & Error Handling
- **Non-extractable / Scanned PDFs**: Detects empty text extraction and alerts the user cleanly.
- **LLM Output Formatting**: Uses regex cleanup and Pydantic schema validation to recover JSON array outputs even if wrapped in markdown code fences (` ```json `).
- **Missing API Keys**: Provides friendly UI warning banners before attempting API calls.
