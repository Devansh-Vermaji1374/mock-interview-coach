from typing import List, Dict, Any, Optional
from rag.vector_store import InterviewVectorStore

class RAGInterviewTools:
    """Tools for retrieving questions, benchmark guidelines, and historical feedback."""

    def __init__(self, vector_store: Optional[InterviewVectorStore] = None):
        self.vector_store = vector_store or InterviewVectorStore()

    def retrieve_interview_questions(
        self,
        company: str,
        role: str,
        interview_type: str,
        difficulty: str,
        query: str = "interview question"
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant company or general questions from ChromaDB."""
        results = self.vector_store.search(
            query=f"{company} {role} {interview_type} {difficulty} {query}",
            company=company,
            role=role,
            interview_type=interview_type,
            difficulty=difficulty,
            n_results=3
        )
        if not results:
            # Fallback to general search without company constraint
            results = self.vector_store.search(
                query=f"{role} {interview_type} {query}",
                n_results=3
            )
        return results

    def retrieve_answer_guidelines(self, question: str, company: str) -> str:
        """Retrieve expected key concepts and evaluation guidelines for a question."""
        results = self.vector_store.search(
            query=question,
            company=company,
            n_results=2
        )
        if not results:
            results = self.vector_store.search(query=question, n_results=2)

        context_texts = []
        for r in results:
            meta = r.get("metadata", {})
            source = meta.get("source", "Knowledge Base")
            verified = "Verified Real" if meta.get("verified") else "Curated Benchmark (Synthetic)"
            doc = r.get("document", "")
            context_texts.append(f"[{source} | {verified}]:\n{doc}")

        return "\n\n".join(context_texts) if context_texts else "No specific company benchmark guidelines found in knowledge base."
