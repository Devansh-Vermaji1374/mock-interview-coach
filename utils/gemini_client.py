import os
import json
import re
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

class GeminiLLMService:
    """Production-grade LLM Service utilizing google.genai SDK with gemini-3.5-flash-lite."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-3.5-flash-lite"):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY", "")
        self.model = model
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY is not set.")
        self.client = genai.Client(api_key=self.api_key)

    def generate(self, prompt: str, system_instruction: Optional[str] = None, temperature: float = 0.7) -> str:
        """Generates content via Gemini with error handling."""
        config = types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=system_instruction
        )
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config
            )
            return response.text if response and response.text else ""
        except Exception as e:
            err_msg = str(e)
            print(f"Gemini generation error: {err_msg}")
            if "ResourceExhausted" in err_msg or "429" in err_msg:
                return "API Quota reached. Please pause a moment before submitting your next response."
            return f"Error from AI model: {err_msg}"

    def get_embedding(self, text: str, model: str = "gemini-embedding-001") -> List[float]:
        """Embeds text into a vector using Google Gemini embeddings."""
        try:
            resp = self.client.models.embed_content(
                model=model,
                contents=text
            )
            if hasattr(resp, "embeddings") and resp.embeddings and len(resp.embeddings) > 0:
                return resp.embeddings[0].values
            elif hasattr(resp, "embedding") and resp.embedding:
                return resp.embedding.values
            return []
        except Exception as e:
            print(f"Gemini Embedding error: {e}")
            return []
