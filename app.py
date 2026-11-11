import streamlit as st
import os
import json
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

import config
from utils.gemini_client import GeminiLLMService
from tools.rag_tools import RAGInterviewTools
from rag.pdf_loader import PDFDocumentProcessor
from rag.vector_store import InterviewVectorStore
from agents import (
    IntentRouter,
    InterviewerAgent,
    EvaluatorAgent,
    CoachAgent,
    ReportAgent
)

# Page configuration
st.set_page_config(
    page_title="Company-Smart Mock Interview Coach | Agentic AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #2563EB;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        font-weight: 600;
    }
    .feedback-box {
        background: #F0FDF4;
        border-left: 4px solid #22C55E;
        padding: 10px 14px;
        border-radius: 4px;
        margin-bottom: 10px;
    }
    .weakness-box {
        background: #FEF2F2;
        border-left: 4px solid #EF4444;
        padding: 10px 14px;
        border-radius: 4px;
        margin-bottom: 10px;
    }
    .coach-box {
        background: #EFF6FF;
        border-left: 4px solid #3B82F6;
        padding: 12px 16px;
        border-radius: 6px;
        margin-top: 10px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "evaluations" not in st.session_state:
    st.session_state.evaluations = []
if "current_question" not in st.session_state:
    st.session_state.current_question = ""
if "interview_started" not in st.session_state:
    st.session_state.interview_started = False
if "final_report" not in st.session_state:
    st.session_state.final_report = None
if "latest_coaching" not in st.session_state:
    st.session_state.latest_coaching = None

@st.cache_resource
def get_services():
    llm = GeminiLLMService()
    vector_store = InterviewVectorStore(persist_directory=config.CHROMA_PERSIST_DIRECTORY)
    rag_tools = RAGInterviewTools(vector_store)
    interviewer = InterviewerAgent(llm, rag_tools)
    evaluator = EvaluatorAgent(llm, rag_tools)
    coach = CoachAgent(llm, rag_tools)
    report = ReportAgent(llm)
    router = IntentRouter(llm)
    return llm, vector_store, rag_tools, interviewer, evaluator, coach, report, router

try:
    llm, vector_store, rag_tools, interviewer, evaluator, coach, report_agent, router = get_services()
except Exception as e:
    st.error(f"Initialization Error: {e}. Please ensure GOOGLE_API_KEY is configured in your .env file.")
    st.stop()

# --- Sidebar Configuration ---
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1573496359142-b8d87734a5a2?w=500&auto=format&fit=crop&q=60", use_container_width=True)
    st.title("Interview Setup")
    
    selected_company = st.selectbox(
        "Target Company",
        config.SUPPORTED_COMPANIES,
        index=0,
        help="Select company for benchmark questions and culture expectations"
    )
    
    selected_role = st.selectbox(
        "Job Role",
        config.INTERVIEW_ROLES,
        index=0
    )
    
    interview_type = st.radio(
        "Interview Track",
        config.INTERVIEW_TYPES,
        index=0
    )
    
    difficulty = st.selectbox(
        "Seniority Level",
        config.DIFFICULTY_LEVELS,
        index=1
    )
    
    st.divider()
    
    # Ingestion & Knowledge Base stats
    st.subheader("📚 RAG Knowledge Base")
    stats = vector_store.get_stats()
    st.metric("Total Indexed Questions", stats["total_count"])
    
    # Custom PDF Uploader
    uploaded_pdf = st.file_uploader("Upload Company PDF / Resume", type=["pdf"], help="Expand knowledge base with your own documents")
    if uploaded_pdf is not None:
        if st.button("Ingest Uploaded PDF"):
            with st.spinner("Processing & indexing PDF into ChromaDB..."):
                save_path = config.DATA_DIR / uploaded_pdf.name
                with open(save_path, "wb") as f:
                    f.write(uploaded_pdf.getbuffer())
                
                processor = PDFDocumentProcessor()
                extracted_docs = processor.extract_structured_qa(
                    str(save_path),
                    default_company=selected_company,
                    default_role=selected_role
                )
                vector_store.add_documents(extracted_docs)
                st.success(f"Indexed {len(extracted_docs)} questions from '{uploaded_pdf.name}'!")
                st.rerun()

    st.divider()
    if st.button("Reset / New Session", type="secondary", use_container_width=True):
        st.session_state.messages = []
        st.session_state.evaluations = []
        st.session_state.current_question = ""
        st.session_state.interview_started = False
        st.session_state.final_report = None
        st.session_state.latest_coaching = None
        st.rerun()

# --- Main Dashboard ---
st.markdown('<div class="main-header">🎯 Company-Smart Mock Interview Coach</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-header">Multi-Agent System powered by Google Gemini, ChromaDB RAG, and verified company question benchmarks. Currently targeting: <strong>{selected_company}</strong> ({selected_role})</div>', unsafe_allow_html=True)

# Top Metrics Row
col1, col2, col3, col4 = st.columns(4)
total_evals = len(st.session_state.evaluations)
avg_score = round(sum([e["scores"]["total_score"] for e in st.session_state.evaluations]) / total_evals, 1) if total_evals > 0 else 0.0

with col1:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{total_evals}</div><div class="metric-label">Questions Completed</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{avg_score}/10</div><div class="metric-label">Average Score</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{selected_company}</div><div class="metric-label">Target Organization</div></div>', unsafe_allow_html=True)
with col4:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{difficulty.split()[0]}</div><div class="metric-label">Difficulty Tier</div></div>', unsafe_allow_html=True)

st.write("")

# Action Buttons Bar
b_col1, b_col2, b_col3, b_col4 = st.columns([1.5, 1.5, 1.5, 1.5])
with b_col1:
    start_btn = st.button("🚀 Start Interview", use_container_width=True, disabled=st.session_state.interview_started)
with b_col2:
    coach_btn = st.button("💡 Get Coaching Tips", use_container_width=True, disabled=(total_evals == 0))
with b_col3:
    report_btn = st.button("📊 Generate Final Report", use_container_width=True, disabled=(total_evals == 0), type="primary")
with b_col4:
    show_kb_btn = st.checkbox("🔍 View RAG Sources", value=False)

if start_btn:
    st.session_state.interview_started = True
    with st.spinner(f"Interviewer Agent retrieving {selected_company} questions and initializing..."):
        q_data = interviewer.generate_question(
            company=selected_company,
            role=selected_role,
            interview_type=interview_type,
            difficulty=difficulty,
            history=[],
            user_input="Let's begin the interview."
        )
        first_q = q_data["question_response"]
        st.session_state.current_question = first_q
        st.session_state.messages.append({
            "role": "assistant",
            "content": first_q,
            "sources": q_data.get("sources", [])
        })
    st.rerun()

if coach_btn:
    with st.spinner("Coach Agent synthesizing feedback & performance trends..."):
        coaching_text = coach.provide_coaching(
            company=selected_company,
            role=selected_role,
            evaluations=st.session_state.evaluations,
            session_history=st.session_state.messages
        )
        st.session_state.latest_coaching = coaching_text

if report_btn:
    with st.spinner("Report Agent compiling comprehensive evaluation metrics..."):
        rep_text = report_agent.generate_final_report(
            company=selected_company,
            role=selected_role,
            interview_type=interview_type,
            difficulty=difficulty,
            evaluations=st.session_state.evaluations
        )
        st.session_state.final_report = rep_text

# Live Coaching Banner if requested
if st.session_state.latest_coaching:
    st.markdown('<div class="coach-box">', unsafe_allow_html=True)
    st.markdown("### 💡 Live Agentic Coaching Insights")
    st.markdown(st.session_state.latest_coaching)
    st.markdown('</div>', unsafe_allow_html=True)

# Final Report Display if generated
if st.session_state.final_report:
    st.success("Final Interview Report Ready!")
    with st.expander("📄 View Full Comprehensive Performance Report", expanded=True):
        st.markdown(st.session_state.final_report)
        st.download_button(
            label="📥 Download Report (.md)",
            data=st.session_state.final_report,
            file_name=f"Interview_Report_{selected_company}_{selected_role.replace(' ', '_')}.md",
            mime="text/markdown"
        )

# Main Interview Chat Layout
chat_col, eval_col = st.columns([1.6, 1.1])

with chat_col:
    st.subheader("💬 Interview Conversation")
    
    # Display Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if show_kb_btn and "sources" in msg and msg["sources"]:
                st.caption(f"📍 Grounded Sources: {', '.join(msg['sources'])}")

    # Chat Input Box
    user_input = st.chat_input("Type your response to the interviewer...")
    if user_input:
        if not st.session_state.interview_started:
            st.session_state.interview_started = True

        # 1. Record User Message
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # 2. Intelligent Routing Check
        route = router.route_intent(user_input, "interview")
        
        if route == "report":
            with st.spinner("Report Agent compiling your session..."):
                st.session_state.final_report = report_agent.generate_final_report(
                    selected_company, selected_role, interview_type, difficulty, st.session_state.evaluations
                )
            st.rerun()
        elif route == "coach":
            with st.spinner("Coach Agent preparing customized recommendations..."):
                st.session_state.latest_coaching = coach.provide_coaching(
                    selected_company, selected_role, st.session_state.evaluations, st.session_state.messages
                )
            st.rerun()
        else:
            # 3. Evaluate the candidate's answer if there was a previous question
            if st.session_state.current_question:
                with st.spinner("Evaluator Agent scoring your response..."):
                    eval_res = evaluator.evaluate_answer(
                        company=selected_company,
                        role=selected_role,
                        question=st.session_state.current_question,
                        candidate_answer=user_input
                    )
                    eval_res["question"] = st.session_state.current_question
                    eval_res["answer"] = user_input
                    st.session_state.evaluations.append(eval_res)

            # 4. Generate next interviewer question / follow-up
            with st.spinner(f"Interviewer Agent analyzing answer & framing next question..."):
                next_q_data = interviewer.generate_question(
                    company=selected_company,
                    role=selected_role,
                    interview_type=interview_type,
                    difficulty=difficulty,
                    history=st.session_state.messages,
                    user_input=user_input
                )
                next_q = next_q_data["question_response"]
                st.session_state.current_question = next_q
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": next_q,
                    "sources": next_q_data.get("sources", [])
                })

            st.rerun()

# Evaluation & Performance Sidebar Column
with eval_col:
    st.subheader("⚡ Real-Time Answer Evaluation")
    
    if st.session_state.evaluations:
        latest_eval = st.session_state.evaluations[-1]
        sc = latest_eval.get("scores", {})
        
        st.markdown(f"#### Question {len(st.session_state.evaluations)} Score: **{sc.get('total_score', 'N/A')}/10**")
        
        # Criterion Progress Bars
        col_c, col_d = st.columns(2)
        with col_c:
            st.metric("Correctness", f"{sc.get('correctness', 0)}/10")
            st.metric("Relevance", f"{sc.get('relevance', 0)}/10")
        with col_d:
            st.metric("Tech Depth", f"{sc.get('technical_depth', 0)}/10")
            st.metric("Clarity", f"{sc.get('communication_clarity', 0)}/10")

        st.markdown("##### 🌟 Key Strengths")
        for s in latest_eval.get("strengths", []):
            st.markdown(f'<div class="feedback-box">✔️ {s}</div>', unsafe_allow_html=True)
            
        st.markdown("##### ⚠️ Improvement Opportunities")
        for w in latest_eval.get("weaknesses", []):
            st.markdown(f'<div class="weakness-box">💡 {w}</div>', unsafe_allow_html=True)

        with st.expander("📖 Ideal Answer Concepts Benchmark"):
            st.info(latest_eval.get("ideal_answer_summary", "Review system architecture and trade-offs."))

        # Score History Table
        if len(st.session_state.evaluations) > 1:
            st.markdown("##### 📈 Session Score Progression")
            history_rows = []
            for i, e in enumerate(st.session_state.evaluations, 1):
                s = e.get("scores", {})
                history_rows.append({
                    "Q#": f"Q{i}",
                    "Total": s.get("total_score", 0),
                    "Correct": s.get("correctness", 0),
                    "Depth": s.get("technical_depth", 0),
                    "Clarity": s.get("communication_clarity", 0)
                })
            df_hist = pd.DataFrame(history_rows)
            st.dataframe(df_hist, hide_index=True, use_container_width=True)
    else:
        st.info("Answer an interview question in the chat to see real-time 4-criterion evaluation and strength/weakness analysis here.")
