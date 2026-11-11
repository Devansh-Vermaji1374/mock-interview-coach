from typing import List, Dict, Any
from utils.gemini_client import GeminiLLMService
from tools.rag_tools import RAGInterviewTools
from prompts.interview_prompts import COACH_PROMPT

class CoachAgent:
    """Coach Agent: Analyzes session trends, recurring weaknesses, and provides targeted tips."""

    def __init__(self, llm_service: GeminiLLMService, rag_tools: RAGInterviewTools):
        self.llm_service = llm_service
        self.rag_tools = rag_tools

    def provide_coaching(
        self,
        company: str,
        role: str,
        evaluations: List[Dict[str, Any]],
        session_history: List[Dict[str, str]]
    ) -> str:
        """Generates real-time targeted coaching recommendations."""
        if not evaluations:
            return "Start answering interview questions to receive live, personalized coaching suggestions!"

        # Summarize evaluations
        eval_summary_lines = []
        for i, ev in enumerate(evaluations, 1):
            scores = ev.get("scores", {})
            strengths = ", ".join(ev.get("strengths", []))
            weaknesses = ", ".join(ev.get("weaknesses", []))
            eval_summary_lines.append(
                f"Question {i} | Overall: {scores.get('total_score', 'N/A')}/10 | "
                f"Correctness: {scores.get('correctness')}, Depth: {scores.get('technical_depth')}, "
                f"Relevance: {scores.get('relevance')}, Clarity: {scores.get('communication_clarity')}\n"
                f"Strengths: {strengths}\nWeaknesses: {weaknesses}"
            )

        eval_summary = "\n\n".join(eval_summary_lines)

        # Retrieve historical feedback references from RAG
        sample_query = "interview feedback preparation tips " + company
        retrieved_feedback = self.rag_tools.retrieve_answer_guidelines(sample_query, company)

        history_summary = f"Total turns completed: {len(evaluations)}. Target: {company} ({role})"

        prompt = COACH_PROMPT.format(
            company=company,
            role=role,
            session_summary=history_summary,
            evaluations_summary=eval_summary,
            retrieved_feedback=retrieved_feedback
        )

        return self.llm_service.generate(prompt, temperature=0.6)
