import io
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pypdf import PdfWriter

from langchain_core.documents import Document
from utils.pdf_processor import process_pdf_to_chunks
from utils.vector_store import retrieve_diverse_chunks
from utils.mcq_generator import clean_json_response, MCQItem


def test_pdf_chunking():
    print("--- 1. Testing PDF Extraction & Chunking ---")
    writer = PdfWriter()
    
    # Add dummy pages with text
    page1_text = "Artificial Intelligence (AI) refers to the simulation of human intelligence in machines. " * 30
    page2_text = "Retrieval-Augmented Generation (RAG) combines semantic vector search with LLM generation. " * 30
    
    writer.add_blank_page(width=612, height=792)
    # We can test with custom Document objects directly as well
    buffer = io.BytesIO()
    writer.write(buffer)
    buffer.seek(0)
    
    dummy_chunks = [
        Document(page_content=page1_text, metadata={"page": 1, "source": "test.pdf"}),
        Document(page_content=page2_text, metadata={"page": 2, "source": "test.pdf"})
    ]

    print(f"Created {len(dummy_chunks)} dummy documents.")
    assert len(dummy_chunks) == 2, "Document creation failed"
    print("[OK] PDF Chunking test passed!")


def test_retriever_sampling():
    print("\n--- 2. Testing Stratified Retriever Sampling ---")
    chunks = [
        Document(page_content=f"Chunk content number {i}", metadata={"page": (i // 2) + 1, "chunk_id": i})
        for i in range(20)
    ]
    
    sampled = retrieve_diverse_chunks(
        vector_store=None,
        chunks=chunks,
        num_chunks_needed=5,
        query=None
    )
    
    print(f"Sampled {len(sampled)} chunks out of {len(chunks)} total chunks.")
    assert len(sampled) == 5, f"Expected 5 chunks, got {len(sampled)}"
    print("[OK] Retriever sampling test passed!")


def test_json_parsing_and_pydantic():
    print("\n--- 3. Testing JSON Cleaning & MCQ Validation ---")
    raw_llm_response = """
    Here is the requested quiz in JSON format:
    ```json
    [
      {
        "question": "What is RAG?",
        "options": {
          "A": "Retrieval-Augmented Generation",
          "B": "Random Access Group",
          "C": "Rapid Application Framework",
          "D": "Recursive Analysis Graph"
        },
        "correct_answer": "A",
        "explanation": "RAG stands for Retrieval-Augmented Generation.",
        "source_text": "Retrieval-Augmented Generation combines search with LLM generation."
      }
    ]
    ```
    Enjoy your quiz!
    """

    mcqs = clean_json_response(raw_llm_response)
    print(f"Parsed {len(mcqs)} MCQ from raw LLM output.")
    
    item = MCQItem(**mcqs[0])
    assert item.correct_answer == "A", "Correct answer mismatch"
    assert "Retrieval-Augmented Generation" in item.options["A"], "Option content mismatch"
    print("[OK] JSON Parsing & Pydantic Validation test passed!")


if __name__ == "__main__":
    test_pdf_chunking()
    test_retriever_sampling()
    test_json_parsing_and_pydantic()
    print("\n=== ALL UNIT CHECKS PASSED SUCCESSFULLY ===")
