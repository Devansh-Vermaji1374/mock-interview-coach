from typing import List, Dict, Any
from utils.gemini_client import GeminiLLMService
from prompts.interview_prompts import REPORT_PROMPT

class ReportAgent:
    """Report Agent: Synthesizes entire interview session into a formal, comprehensive performance report."""

    def __init__(self, llm_service: GeminiLLMService):
        self.llm_service = llm_service

    def generate_final_report(
        self,
        company: str,
        role: str,
        interview_type: str,
        difficulty: str,
        evaluations: List[Dict[str, Any]]
    ) -> str:
        """Generates comprehensive final report with aggregate scores and recommendations."""
        if not evaluations:
            return "No interview questions have been evaluated yet. Please answer at least one question to generate a report."

        # Compile detailed transcript
        details = []
        for i, ev in enumerate(evaluations, 1):
            sc = ev.get("scores", {})
            q = ev.get("question", "N/A")
            ans = ev.get("answer", "N/A")
            str_list = "\n  - ".join(ev.get("strengths", ["N/A"]))
            wk_list = "\n  - ".join(ev.get("weaknesses", ["N/A"]))
            
            details.append(
                f"### Question {i}: {q}\n"
                f"- **Candidate Answer**: {ans}\n"
                f"- **Scores**: Total: {sc.get('total_score', 'N/A')}/10 "
                f"[Correctness: {sc.get('correctness')}/10, Depth: {sc.get('technical_depth')}/10, "
                f"Relevance: {sc.get('relevance')}/10, Clarity: {sc.get('communication_clarity')}/10]\n"
                f"- **Key Strengths**:\n  - {str_list}\n"
                f"- **Areas Missed / Weaknesses**:\n  - {wk_list}\n"
            )

        transcript_text = "\n".join(details)

        prompt = REPORT_PROMPT.format(
            company=company,
            role=role,
            interview_type=interview_type,
            difficulty=difficulty,
            detailed_transcript=transcript_text
        )

        return self.llm_service.generate(prompt, temperature=0.3)
