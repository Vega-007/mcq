import random
from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS


def get_embedding_model(provider: str, api_key: str):
    """
    Returns appropriate embedding model based on provider ('gemini' or 'openai').
    """
    clean_key = api_key.strip() if api_key else ""
    if provider.lower() == "gemini":
        if clean_key:
            import os
            os.environ["GOOGLE_API_KEY"] = clean_key
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=clean_key
        )
    elif provider.lower() == "openai":
        if clean_key:
            import os
            os.environ["OPENAI_API_KEY"] = clean_key
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=clean_key
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def create_vector_store(chunks: List[Document], provider: str, api_key: str) -> FAISS:
    """
    Creates an in-memory FAISS vector store from semantic document chunks.
    """
    if not chunks:
        raise ValueError("Cannot create vector store from empty chunks.")

    embeddings = get_embedding_model(provider=provider, api_key=api_key)
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store


def retrieve_diverse_chunks(
    vector_store: FAISS,
    chunks: List[Document],
    num_chunks_needed: int = 5,
    query: Optional[str] = None
) -> List[Document]:
    """
    Retrieves representative, diverse text chunks from the vector database for RAG MCQ generation.
    Combines Maximal Marginal Relevance (MMR) search with stratified chunk selection across the document.
    """
    total_chunks = len(chunks)
    if total_chunks <= num_chunks_needed:
        return chunks

    retrieved = []
    
    if query:
        # Use MMR search to fetch diverse relevant chunks based on user query
        retrieved = vector_store.max_marginal_relevance_search(
            query=query,
            k=num_chunks_needed,
            fetch_k=min(total_chunks, num_chunks_needed * 3)
        )
    else:
        # Stratified sampling across the document timeline + MMR for concept diversity
        step = total_chunks / num_chunks_needed
        indices = [int(i * step) for i in range(num_chunks_needed)]
        retrieved = [chunks[i] for i in indices if i < total_chunks]

    return retrieved
