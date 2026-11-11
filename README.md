# Company-Smart Mock Interview Coach – Agentic AI

An AI-powered mock interview preparation system leveraging company-specific interview questions, historical interview data, Retrieval-Augmented Generation (RAG) with ChromaDB, and a collaborative Multi-Agent Architecture powered by Google Gemini.

---

## 🌟 Key Features

1. **Company & Role Customization**: Select target companies (Google, Amazon, Microsoft, Infosys, TCS, etc.), job roles (SDE, Frontend, Backend, AI/ML), difficulty tiers, and interview formats (Technical vs HR/Behavioral).
2. **Multi-Agent Collaboration**:
   - **Interviewer Agent**: Asks realistic, grounded company questions one at a time and creates contextual follow-ups.
   - **Evaluator Agent**: Evaluates responses across 4 criteria (Correctness, Technical Depth, Relevance, Communication Clarity) with 1-10 scores and strengths/weaknesses.
   - **Coach Agent**: Identifies recurring patterns and recommends targeted preparation topics.
   - **Report Agent**: Synthesizes the interview into an executive performance report with hiring recommendations and action plans.
   - **Router Agent**: Analyzes candidate intent and dynamically routes requests.
3. **RAG Knowledge Base & Persistence**:
   - Automatic PDF ingestion (`pypdf`) with question and answer extraction.
   - Vector persistence with **ChromaDB**.
   - Embeddings powered by Google Gemini Embeddings.
   - Dynamic user PDF upload capability in the UI.
4. **Clean Streamlit UI**:
   - Split-screen layout: interactive live interview chat on the left, instant evaluation scorecard and progression metrics on the right.
   - Collapsible benchmark criteria, RAG provenance badges, and download report options.

---

## 📂 Project Architecture

```
mock_interview_coach/
├── .env.example              # Environment variables template
├── .env                      # Local environment configuration (with GOOGLE_API_KEY)
├── requirements.txt          # Python dependencies
├── config.py                 # System configuration & supported parameters
├── app.py                    # Streamlit frontend & interactive dashboard
├── agents/                   # Multi-Agent implementations
│   ├── __init__.py
│   ├── router_agent.py       # Intent router
│   ├── interviewer_agent.py  # Adaptive company questioner
│   ├── evaluator_agent.py    # 4-criterion answer evaluator
│   ├── coach_agent.py        # Real-time personalized coach
│   └── report_agent.py       # Comprehensive report generator
├── rag/                      # RAG Knowledge Pipeline
│   ├── __init__.py
│   ├── pdf_loader.py         # PDF text extractor & structured Q&A parser
│   ├── vector_store.py       # Persistent ChromaDB vector store
│   └── ingest.py             # Knowledge ingestion & indexing script
├── tools/                    # LangChain/Agent tools
│   ├── __init__.py
│   └── rag_tools.py          # Grounded question & guideline retrieval
├── prompts/                  # Structured prompt templates
│   ├── __init__.py
│   └── interview_prompts.py  # Prompts for all agents
├── data/                     # Raw datasets & documents
│   ├── 30-SOFTWARE-ENGINEERING-Interview-Questions-and-Answers_FINAL.pdf
│   └── sample_company_data.json
├── database/                 # ChromaDB persistence storage
│   └── chroma/
└── utils/                    # Helper services
    ├── __init__.py
    └── gemini_client.py      # Google GenAI LLM & Embeddings service
```

---

## 🚀 Setup & Local Execution Instructions

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.12)
- Google Gemini API Key from [Google AI Studio](https://aistudio.google.com/)

### 2. Environment Configuration
Clone or navigate to the project directory:
```bash
cd mock_interview_coach
```

Ensure your `.env` file exists and has your Gemini API key:
```env
GOOGLE_API_KEY=your_actual_gemini_api_key_here
CHROMA_PERSIST_DIRECTORY=database/chroma
DEFAULT_MODEL=gemini-3.8-flash
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Ingest Initial Knowledge Base
Ingest the included Software Engineering PDF and company benchmark datasets into ChromaDB:
```bash
python -m rag.ingest
```

### 5. Launch the Application
Run the Streamlit frontend:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🧪 Testing the Multi-Agent Pipeline

To test the agents directly in the console:
```bash
python -c "from rag.vector_store import InterviewVectorStore; print('Indexed:', InterviewVectorStore().get_stats())"
```
