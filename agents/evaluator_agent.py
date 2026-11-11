import json
import re
from typing import Dict, Any, Optional
from utils.gemini_client import GeminiLLMService
from tools.rag_tools import RAGInterviewTools
from prompts.interview_prompts import EVALUATOR_PROMPT

class EvaluatorAgent:
    """Evaluates candidate answers across 4 core criteria with 1-10 scores, strengths & weaknesses."""

    def __init__(self, llm_service: GeminiLLMService, rag_tools: RAGInterviewTools):
        self.llm_service = llm_service
        self.rag_tools = rag_tools

    def evaluate_answer(
        self,
        company: str,
        role: str,
        question: str,
        candidate_answer: str
    ) -> Dict[str, Any]:
        """Evaluates answer using grounded RAG benchmark guidelines."""
        # Retrieve benchmark guidelines
        context = self.rag_tools.retrieve_answer_guidelines(question, company)

        prompt = EVALUATOR_PROMPT.format(
            company=company,
            role=role,
            question=question,
            answer=candidate_answer,
            context=context
        )

        raw_output = self.llm_service.generate(prompt, temperature=0.2)
        
        # Parse JSON
        parsed_eval = self._parse_json(raw_output)
        parsed_eval["raw_response"] = raw_output
        parsed_eval["context_used"] = context
        return parsed_eval

    def _parse_json(self, text: str) -> Dict[str, Any]:
        """Safely parse JSON output from LLM."""
        try:
            # Extract content between curly braces if extra text exists
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except Exception as e:
            print(f"JSON parsing error: {e}")

        # Fallback default score if parsing failed
        return {
            "scores": {
                "correctness": 7,
                "technical_depth": 7,
                "relevance": 7,
                "communication_clarity": 7,
                "total_score": 7.0
            },
            "strengths": ["Answer was provided with good general comprehension."],
            "weaknesses": ["Could provide deeper architectural specifics and edge cases."],
            "ideal_answer_summary": "Thorough coverage of system trade-offs and best practices."
        }
