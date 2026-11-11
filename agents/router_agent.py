import re
from typing import Dict, Any
from utils.gemini_client import GeminiLLMService

class IntentRouter:
    """Classifies user input or application state to route to the appropriate agent."""

    def __init__(self, llm_service: GeminiLLMService):
        self.llm_service = llm_service

    def route_intent(self, user_message: str, current_phase: str) -> str:
        """
        Routes intent to one of:
        - 'report': if user requests final report, conclusion, or summary
        - 'coach': if user asks for advice, guidance, tips, or what to improve
        - 'evaluate': if candidate is submitting an answer to an active question
        - 'interview': default practice/next question flow
        """
        clean_msg = user_message.lower().strip()
        
        # Fast rule-based overrides
        if any(w in clean_msg for w in ["generate report", "final report", "show my report", "finish interview", "end interview"]):
            return "report"
            
        if any(w in clean_msg for w in ["coach me", "how can i improve", "give me tips", "what are my weaknesses", "coaching advice"]):
            return "coach"

        # Prompt-based routing if ambiguous
        routing_prompt = f"""Classify the user intent into exactly ONE of the following categories:
- 'interview': The user is answering an interview question, continuing the interview, or asking for the next question.
- 'evaluate': The user is requesting immediate feedback or scoring on their answer.
- 'coach': The user is asking for career advice, coaching suggestions, or general improvement tips.
- 'report': The user wants to end the interview, generate the final evaluation report, or see overall performance.

User Input: "{user_message}"
Current Phase: "{current_phase}"

Respond with ONLY the category name: interview, evaluate, coach, or report.
"""
        response = self.llm_service.generate(routing_prompt, temperature=0.0).strip().lower()
        for cat in ["report", "coach", "evaluate", "interview"]:
            if cat in response:
                return cat
        return "interview"
