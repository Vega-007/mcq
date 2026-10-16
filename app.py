import os
import json
import streamlit as st
from dotenv import load_dotenv

# Import custom utilities
from utils.pdf_processor import process_pdf_to_chunks
from utils.vector_store import create_vector_store, retrieve_diverse_chunks
from utils.mcq_generator import generate_mcqs_from_chunks

# Load environment variables
load_dotenv()

# Page Configuration
st.set_page_config(
    page_title="Intelligent MCQ Generator (RAG)",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics
st.markdown("""
<style>
    /* Main Theme & Custom Card Styling */
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 50%, #EC4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #64748B;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .stCard {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        border: 1px solid #E2E8F0;
        margin-bottom: 20px;
    }
    .badge-success {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 4px 12px;
        border-radius: 16px;
        font-weight: 600;
        font-size: 0.88rem;
    }
    .badge-danger {
        background-color: #FDE8E8;
        color: #9B1C1C;
        padding: 4px 12px;
        border-radius: 16px;
        font-weight: 600;
        font-size: 0.88rem;
    }
    .badge-info {
        background-color: #E1EFFE;
        color: #1E429F;
        padding: 4px 12px;
        border-radius: 16px;
        font-weight: 600;
        font-size: 0.88rem;
    }
    .source-box {
        background-color: #F8FAFC;
        border-left: 4px solid #6366F1;
        padding: 12px 16px;
        border-radius: 4px;
        font-size: 0.9rem;
        color: #334155;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initializes Streamlit session state variables."""
    if "chunks" not in st.session_state:
        st.session_state.chunks = None
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "pdf_name" not in st.session_state:
        st.session_state.pdf_name = None
    if "mcqs" not in st.session_state:
        st.session_state.mcqs = None
    if "quiz_submitted" not in st.session_state:
        st.session_state.quiz_submitted = False
    if "user_answers" not in st.session_state:
        st.session_state.user_answers = {}
    if "score" not in st.session_state:
        st.session_state.score = 0


def main():
    init_session_state()

    # Sidebar Layout
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3407/3407024.png", width=60)
        st.title("⚙️ RAG Configuration")
        st.markdown("---")

        # 1. API Provider & Key Selection
        api_provider = st.selectbox(
            "Select LLM Provider",
            options=["Gemini", "OpenAI"],
            index=0,
            help="Google Gemini is recommended."
        )

        default_key = os.getenv("GOOGLE_API_KEY") if api_provider == "Gemini" else os.getenv("OPENAI_API_KEY")
        api_key = st.text_input(
            f"Enter {api_provider} API Key",
            value=default_key or "",
            type="password",
            help="Type or paste your API key here and press Enter."
        )

        api_key = api_key.strip() if api_key else ""

        if api_key:
            if api_provider == "Gemini":
                os.environ["GOOGLE_API_KEY"] = api_key
            else:
                os.environ["OPENAI_API_KEY"] = api_key
            st.caption("✅ **API Key Active** for current session")
        else:
            st.warning(f"⚠️ Please enter your {api_provider} API Key above and press Enter.")

        st.markdown("---")
        st.subheader("📋 Quiz Settings")
        
        num_questions = st.select_slider(
            "Number of Questions",
            options=[3, 5, 10, 15, 20],
            value=5
        )

        difficulty = st.radio(
            "Difficulty Level",
            options=["Easy", "Medium", "Hard"],
            index=1,
            horizontal=True
        )

        topic_query = st.text_input(
            "Optional Focus Concept / Topic",
            placeholder="e.g. Neural Networks, Photosynthesis...",
            help="Leave blank to cover the entire PDF document evenly."
        )

        st.markdown("---")
        st.subheader("📄 Upload Document")
        uploaded_pdf = st.file_uploader(
            "Upload Unit / Chapter PDF",
            type=["pdf"],
            help="Upload a text-based PDF chapter to build the RAG knowledge base."
        )

        if uploaded_pdf:
            if not api_key:
                st.warning("🔑 Please enter your API Key above to enable PDF processing.")
            elif st.button("🚀 Process PDF & Index RAG", use_container_width=True):
                with st.spinner("Parsing PDF and building FAISS Vector Store..."):
                    try:
                        chunks = process_pdf_to_chunks(uploaded_pdf, chunk_size=1000, chunk_overlap=150)
                        vector_store = create_vector_store(chunks, provider=api_provider, api_key=api_key)

                        st.session_state.chunks = chunks
                        st.session_state.vector_store = vector_store
                        st.session_state.pdf_name = uploaded_pdf.name
                        st.session_state.mcqs = None  # Reset existing quiz
                        st.session_state.quiz_submitted = False
                        st.session_state.user_answers = {}

                        st.success(f"✅ Indexed {len(chunks)} text chunks into FAISS Vector DB!")
                    except Exception as e:
                        err_str = str(e)
                        st.error(f"❌ Error processing PDF: {err_str}")
                        if "API_KEY_INVALID" in err_str or "API key not valid" in err_str or "400" in err_str:
                            st.warning(
                                "💡 **Invalid API Key Detected**:\n"
                                "1. Ensure you copied the full key without extra quotes or characters.\n"
                                "2. For Google Gemini, create a free API key at [Google AI Studio](https://aistudio.google.com/).\n"
                                "3. Make sure the Generative Language API is enabled for your project."
                            )

        st.markdown("---")
        st.caption("Built with LangChain • FAISS • Streamlit • Gemini/OpenAI")

    # Main UI
    st.markdown('<div class="main-title">🧠 Intelligent MCQ Generator from PDF</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Retrieval-Augmented Generation (RAG) system for conceptual quiz creation with automated explanations and source traceability.</div>',
        unsafe_allow_html=True
    )

    # Document Status Bar
    if st.session_state.pdf_name and st.session_state.chunks:
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.info(f"📄 **Active PDF**: {st.session_state.pdf_name}")
        with col_b:
            st.info(f"🧩 **Total Chunks**: {len(st.session_state.chunks)}")
        with col_c:
            st.success("⚡ **FAISS Index Status**: Ready")
    else:
        st.info("👈 Upload a PDF and provide your API Key in the sidebar to get started.")

    st.markdown("---")

    # Action Area: Generate Quiz
    if st.session_state.vector_store and st.session_state.chunks:
        if st.session_state.mcqs is None:
            if st.button("✨ Generate MCQs Now", type="primary", use_container_width=True):
                if not api_key:
                    st.error("Please enter your API key in the sidebar.")
                else:
                    with st.spinner("Retrieving semantic chunks and crafting conceptual MCQs..."):
                        try:
                            # 1. Retrieve relevant / diverse chunks via RAG retriever
                            target_query = topic_query.strip() if topic_query else None
                            retrieved_chunks = retrieve_diverse_chunks(
                                vector_store=st.session_state.vector_store,
                                chunks=st.session_state.chunks,
                                num_chunks_needed=num_questions,
                                query=target_query
                            )

                            # 2. Generate MCQs using LangChain pipeline
                            mcqs = generate_mcqs_from_chunks(
                                chunks=retrieved_chunks,
                                provider=api_provider,
                                api_key=api_key,
                                num_questions=num_questions,
                                difficulty=difficulty
                            )

                            st.session_state.mcqs = mcqs
                            st.session_state.quiz_submitted = False
                            st.session_state.user_answers = {}
                            st.rerun()

                        except Exception as e:
                            st.error(f"❌ Failed to generate MCQs: {str(e)}")

    # Display Quiz Form & Results
    if st.session_state.mcqs:
        mcqs = st.session_state.mcqs
        st.subheader(f"📝 Interactive Quiz ({len(mcqs)} Questions - {difficulty} Level)")

        # Quiz Submission Handler
        if not st.session_state.quiz_submitted:
            with st.form("quiz_form"):
                for idx, item in enumerate(mcqs):
                    st.markdown(f"#### **Q{idx + 1}. {item['question']}**")
                    options = item["options"]
                    
                    # Formatted options for display
                    choice_labels = [f"{k}) {v}" for k, v in options.items()]
                    
                    user_choice = st.radio(
                        label=f"Select your answer for Question {idx + 1}:",
                        options=choice_labels,
                        index=None,
                        key=f"q_{idx}"
                    )
                    
                    # Store option key (A, B, C, or D)
                    if user_choice:
                        selected_key = user_choice.split(")")[0].strip()
                        st.session_state.user_answers[idx] = selected_key
                    st.markdown("---")

                submit_button = st.form_submit_button("✅ Submit Quiz & View Score", type="primary", use_container_width=True)

                if submit_button:
                    # Calculate Score
                    score = 0
                    for idx, item in enumerate(mcqs):
                        if st.session_state.user_answers.get(idx) == item["correct_answer"]:
                            score += 1
                    
                    st.session_state.score = score
                    st.session_state.quiz_submitted = True
                    st.rerun()

        else:
            # Score Overview Card
            total_q = len(mcqs)
            score = st.session_state.score
            percentage = (score / total_q) * 100

            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1E1B4B 0%, #312E81 100%); padding: 25px; border-radius: 12px; color: white; text-align: center; margin-bottom: 30px;">
                <h2 style="margin:0; font-size: 2rem;">📊 Assessment Summary</h2>
                <h1 style="margin: 10px 0; font-size: 3.2rem; color: #818CF8;">{score} / {total_q} ({percentage:.0f}%)</h1>
                <p style="margin:0; font-size: 1.1rem; color: #C7D2FE;">{"🎉 Outstanding performance!" if percentage >= 80 else "👍 Good attempt! Review the detailed solutions below."}</p>
            </div>
            """, unsafe_allow_html=True)

            # Questions Assessment Breakdown
            for idx, item in enumerate(mcqs):
                user_ans = st.session_state.user_answers.get(idx, "Unanswered")
                correct_ans = item["correct_answer"]
                is_correct = (user_ans == correct_ans)

                st.markdown(f"#### **Q{idx + 1}. {item['question']}**")

                if is_correct:
                    st.markdown(f'<span class="badge-success">✔ Correct Answer</span>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<span class="badge-danger">✘ Incorrect (Your choice: {user_ans})</span>', unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Render Options with Color Highlights
                for k, v in item["options"].items():
                    if k == correct_ans:
                        st.markdown(f"🟢 **{k}) {v}** *(Correct Answer)*")
                    elif k == user_ans and not is_correct:
                        st.markdown(f"🔴 **{k}) {v}** *(Your Selection)*")
                    else:
                        st.markdown(f"⚪ {k}) {v}")

                # Explanation Block
                st.markdown(f"**💡 Explanation:** {item['explanation']}")

                # Source Traceability Accordion
                with st.expander("🔍 View Source Reference Excerpt (RAG Traceability)"):
                    st.markdown(f'<div class="source-box">"{item.get("source_text", "Source chunk reference unavailable.")}"</div>', unsafe_allow_html=True)

                st.markdown("---")

            # Export & Action Buttons
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Export Quiz JSON
                json_data = json.dumps(mcqs, indent=2)
                st.download_button(
                    label="📥 Export Quiz (JSON)",
                    data=json_data,
                    file_name=f"mcq_quiz_{st.session_state.pdf_name}.json",
                    mime="application/json",
                    use_container_width=True
                )
                
            with col2:
                # Export Answer Key & Explanations Text
                txt_output = f"MCQ Quiz & Answer Key\nSource PDF: {st.session_state.pdf_name}\nScore: {score}/{total_q}\n\n"
                for idx, item in enumerate(mcqs):
                    txt_output += f"Q{idx+1}: {item['question']}\n"
                    for k, v in item['options'].items():
                        txt_output += f"  {k}) {v}\n"
                    txt_output += f"Correct Answer: {item['correct_answer']}\n"
                    txt_output += f"Explanation: {item['explanation']}\n\n"

                st.download_button(
                    label="📄 Export Answer Key (TXT)",
                    data=txt_output,
                    file_name=f"mcq_answers_{st.session_state.pdf_name}.txt",
                    mime="text/plain",
                    use_container_width=True
                )

            with col3:
                if st.button("🔄 Retake / Regenerate Quiz", use_container_width=True):
                    st.session_state.mcqs = None
                    st.session_state.quiz_submitted = False
                    st.session_state.user_answers = {}
                    st.rerun()


if __name__ == "__main__":
    main()
