import io
from typing import List, Dict, Any
from pypdf import PdfReader
from langchain_core.documents import Document
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter



def extract_text_from_pdf(pdf_file: Any) -> List[Dict[str, Any]]:
    """
    Extracts text page-by-page from an uploaded PDF file stream or path.
    Returns a list of dicts containing 'page_number' and 'text'.
    """
    pages_data = []
    
    try:
        # Handle Streamlit UploadedFile or file path / bytes
        if hasattr(pdf_file, "read"):
            pdf_bytes = pdf_file.read()
            pdf_file.seek(0)  # Reset buffer pointer
            reader = PdfReader(io.BytesIO(pdf_bytes))
        else:
            reader = PdfReader(pdf_file)

        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                pages_data.append({
                    "page_number": i + 1,
                    "text": text.strip()
                })
                
    except Exception as e:
        raise ValueError(f"Failed to parse PDF file: {str(e)}")

    if not pages_data:
        raise ValueError("No extractable text found in the PDF. The file might be scanned images or empty.")

    return pages_data


def process_pdf_to_chunks(
    pdf_file: Any,
    chunk_size: int = 1000,
    chunk_overlap: int = 150
) -> List[Document]:
    """
    Extracts text from PDF and splits it into semantic chunks using RecursiveCharacterTextSplitter.
    Returns a list of LangChain Document objects.
    """
    pages_data = extract_text_from_pdf(pdf_file)
    
    # Create base documents with page metadata
    raw_documents = [
        Document(
            page_content=data["text"],
            metadata={
                "page": data["page_number"],
                "source": getattr(pdf_file, "name", "uploaded_pdf")
            }
        )
        for data in pages_data
    ]

    # Initialize text splitter as required (chunk_size=1000, chunk_overlap=150)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    # Split documents into smaller chunks
    chunks = text_splitter.split_documents(raw_documents)
    
    # Add chunk index metadata for traceability
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    return chunks
