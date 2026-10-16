import json
import re
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document


class MCQItem(BaseModel):
    question: str = Field(description="Clear, conceptual question grounded strictly in the provided text.")
    options: Dict[str, str] = Field(description="Dictionary of 4 distinct choices: A, B, C, D.")
    correct_answer: str = Field(description="The correct option key: 'A', 'B', 'C', or 'D'.")
    explanation: str = Field(description="Detailed reasoning explaining why the correct option is right and others are wrong.")
    source_text: str = Field(description="Exact excerpt or summary of the source chunk used for traceability.")


class MCQResponse(BaseModel):
    mcqs: List[MCQItem]


MCQ_PROMPT_TEMPLATE = """You are an expert educational assessment creator. Your task is to generate {num_questions} high-quality, conceptual Multiple-Choice Questions (MCQs) strictly based on the provided text chunks retrieved from a document.

DIFFICULTY LEVEL: {difficulty}

SOURCE CONTEXT:
-------------------
{context}
-------------------

STRICT CONSTRAINTS & RULES:
1. Base every question ONLY on facts and concepts explicitly mentioned in the SOURCE CONTEXT above. Do not introduce outside knowledge or unverified facts.
2. Each question MUST have exactly 4 distinct options labeled 'A', 'B', 'C', and 'D'.
3. One option must be undeniably correct, while the other three options (distractors) should be plausible yet incorrect based on the text.
4. Include a detailed explanation explaining why the correct answer is right.
5. Include a short 'source_text' snippet directly taken from the context that proves the answer.
6. Target difficulty: {difficulty} level.
7. OUTPUT FORMAT: You MUST return ONLY a valid JSON array of objects. Do not include markdown headers, prose, or conversational wrapper outside the JSON.

REQUIRED JSON FORMAT EXAMPLE:
[
  {{
    "question": "What is the primary function of...?",
    "options": {{
      "A": "Option A text",
      "B": "Option B text",
      "C": "Option C text",
      "D": "Option D text"
    }},
    "correct_answer": "A",
    "explanation": "Option A is correct because...",
    "source_text": "Direct quote from source context..."
  }}
]

Generate exactly {num_questions} MCQs now in JSON format:
"""


def get_llm_instance(provider: str, api_key: str, temperature: float = 0.3):
    """
    Returns an instance of LLM based on provider ('gemini' or 'openai').
    """
    clean_key = api_key.strip() if api_key else ""
    if provider.lower() == "gemini":
        if clean_key:
            import os
            os.environ["GOOGLE_API_KEY"] = clean_key
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model="gemini-3.8-flash",
            google_api_key=clean_key,
            temperature=temperature
        )
    elif provider.lower() == "openai":
        if clean_key:
            import os
            os.environ["OPENAI_API_KEY"] = clean_key
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model="gpt-4o-mini",
            openai_api_key=clean_key,
            temperature=temperature
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def clean_json_response(raw_text: Any) -> List[Dict[str, Any]]:
    """
    Cleans LLM response output to reliably extract and parse JSON list.
    Supports str, list of message content blocks, or objects with text/content.
    """
    if isinstance(raw_text, list):
        parts = []
        for p in raw_text:
            if isinstance(p, dict) and "text" in p:
                parts.append(p["text"])
            elif isinstance(p, str):
                parts.append(p)
            elif hasattr(p, "text"):
                parts.append(p.text)
            else:
                parts.append(str(p))
        text = "".join(parts).strip()
    elif isinstance(raw_text, str):
        text = raw_text.strip()
    elif hasattr(raw_text, "content"):
        return clean_json_response(raw_text.content)
    else:
        text = str(raw_text).strip()
    
    # Strip markdown code fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
    text = text.strip()

    # Try direct parsing first
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "mcqs" in data:
            data = data["mcqs"]
        if isinstance(data, list):
            return data
    except Exception:
        pass

    # Find array pattern [...] if surrounded by extraneous text
    match = re.search(r"\[\s*\{.*\}\s*\]", text, re.DOTALL)
    if match:
        extracted = match.group(0)
        try:
            data = json.loads(extracted)
            if isinstance(data, list):
                return data
        except Exception:
            pass

    display_preview = str(raw_text)[:300]
    raise ValueError(f"Could not parse valid MCQ JSON from LLM response.\nRaw Response: {display_preview}")



def generate_mcqs_from_chunks(
    chunks: List[Document],
    provider: str,
    api_key: str,
    num_questions: int = 5,
    difficulty: str = "Medium"
) -> List[Dict[str, Any]]:
    """
    Generates MCQs using RAG pipeline over retrieved context chunks.
    Returns validated list of MCQ dictionaries.
    """
    if not chunks:
        raise ValueError("No context chunks provided for MCQ generation.")

    # Combine chunk texts with page references
    context_blocks = []
    for chunk in chunks:
        page_info = f"[Page {chunk.metadata.get('page', 'Unknown')}]"
        context_blocks.append(f"{page_info}\n{chunk.page_content}")
        
    combined_context = "\n\n---\n\n".join(context_blocks)

    # Initialize LLM
    llm = get_llm_instance(provider=provider, api_key=api_key)

    # Build Prompt
    prompt = PromptTemplate(
        template=MCQ_PROMPT_TEMPLATE,
        input_variables=["num_questions", "difficulty", "context"]
    )

    formatted_prompt = prompt.format(
        num_questions=num_questions,
        difficulty=difficulty,
        context=combined_context
    )

    # Invoke LLM
    response = llm.invoke(formatted_prompt)
    raw_content = response.content if hasattr(response, "content") else str(response)

    # Clean & Parse JSON
    mcqs_raw = clean_json_response(raw_content)

    # Validate each item using Pydantic model
    validated_mcqs = []
    for item in mcqs_raw:
        try:
            # Ensure correct answer is uppercase letter
            correct = str(item.get("correct_answer", "A")).strip().upper()
            if correct not in ["A", "B", "C", "D"]:
                correct = "A"

            mcq_item = MCQItem(
                question=item.get("question", "Question text missing"),
                options=item.get("options", {"A": "Choice A", "B": "Choice B", "C": "Choice C", "D": "Choice D"}),
                correct_answer=correct,
                explanation=item.get("explanation", "No explanation provided."),
                source_text=item.get("source_text", "Source context excerpt unavailable.")
            )
            validated_mcqs.append(mcq_item.model_dump())
        except Exception:
            continue

    if not validated_mcqs:
        raise ValueError("Failed to generate valid MCQs. Please retry with a different question count or check API Key.")

    return validated_mcqs
