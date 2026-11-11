from typing import List, Dict, Any, Optional
from utils.gemini_client import GeminiLLMService
from tools.rag_tools import RAGInterviewTools
from prompts.interview_prompts import INTERVIEWER_PROMPT

class InterviewerAgent:
    """Interviewer Agent: Asks realistic company questions, maintains context, adapts dynamically."""

    def __init__(self, llm_service: GeminiLLMService, rag_tools: RAGInterviewTools):
        self.llm_service = llm_service
        self.rag_tools = rag_tools

    def generate_question(
        self,
        company: str,
        role: str,
        interview_type: str,
        difficulty: str,
        history: List[Dict[str, str]],
        user_input: str
    ) -> Dict[str, Any]:
        """Generates the next question or adaptive follow-up grounded in RAG."""
        # Retrieve relevant question contexts from ChromaDB
        retrieved_docs = self.rag_tools.retrieve_interview_questions(
            company=company,
            role=role,
            interview_type=interview_type,
            difficulty=difficulty,
            query=user_input or "core interview question"
        )

        context_pieces = []
        sources = []
        for doc in retrieved_docs:
            meta = doc.get("metadata", {})
            src = meta.get("source", "PDF / Benchmark")
            ver = "Verified Real" if meta.get("verified") else "Curated Benchmark (Synthetic)"
            context_pieces.append(f"Source [{src} - {ver}]:\n{doc.get('document', '')}")
            sources.append(f"{src} ({ver})")

        rag_context = "\n\n".join(context_pieces) if context_pieces else "General standard software engineering interview topics."

        # Format conversation history
        history_text = ""
        for turn in history[-6:]:  # Keep recent context
            history_text += f"{turn.get('role', 'interviewer').capitalize()}: {turn.get('content', '')}\n"

        prompt = INTERVIEWER_PROMPT.format(
            company=company,
            role=role,
            interview_type=interview_type,
            difficulty=difficulty,
            context=rag_context,
            history=history_text or "No prior questions asked yet.",
            user_input=user_input or "Ready to start the interview."
        )

        response = self.llm_service.generate(prompt, temperature=0.7)
        return {
            "question_response": response,
            "rag_context": rag_context,
            "sources": list(set(sources))
        }
