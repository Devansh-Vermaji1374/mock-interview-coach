import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from utils.gemini_client import GeminiLLMService

class InterviewVectorStore:
    """ChromaDB Vector Store management for interview Q&A and RAG retrieval."""

    def __init__(self, persist_directory: str = "database/chroma"):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        self.client = chromadb.PersistentClient(path=str(self.persist_directory))
        self.collection = self.client.get_or_create_collection(
            name="interview_knowledge_base",
            metadata={"description": "Company-specific and verified software engineering interview Q&A"}
        )
        self.gemini_service = GeminiLLMService()

    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to ChromaDB with generated embeddings and metadata."""
        if not documents:
            return

        ids = []
        texts = []
        metadatas = []
        embeddings = []

        for doc in documents:
            doc_id = str(doc.get("id"))
            text = f"Question: {doc.get('question', '')}\nContent: {doc.get('content', '')}"
            meta = {
                "company": str(doc.get("company", "General Tech")),
                "role": str(doc.get("role", "Software Development Engineer (SDE / SWE)")),
                "type": str(doc.get("type", "Technical Interview")),
                "difficulty": str(doc.get("difficulty", "Mid Level")),
                "source": str(doc.get("source", "PDF / Dataset")),
                "verified": bool(doc.get("verified", False)),
                "question": str(doc.get("question", ""))[:250]
            }
            
            # Generate Gemini embedding
            emb = self.gemini_service.get_embedding(text)
            if not emb:
                # Fallback: simple deterministic length padding if embedding fails
                continue

            ids.append(doc_id)
            texts.append(text)
            metadatas.append(meta)
            embeddings.append(emb)

        if ids:
            self.collection.upsert(
                ids=ids,
                documents=texts,
                metadatas=metadatas,
                embeddings=embeddings
            )
            print(f"Successfully added/updated {len(ids)} documents in ChromaDB collection.")

    def search(
        self,
        query: str,
        company: Optional[str] = None,
        role: Optional[str] = None,
        interview_type: Optional[str] = None,
        difficulty: Optional[str] = None,
        n_results: int = 4
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant interview Q&A with metadata filtering."""
        query_emb = self.gemini_service.get_embedding(query)
        if not query_emb:
            return []

        # Build Chroma where filter
        filters = []
        if company and company != "All" and company != "General Tech":
            filters.append({"company": company})
        if role and role != "All":
            filters.append({"role": role})
        if interview_type and interview_type != "All":
            filters.append({"type": interview_type})
        if difficulty and difficulty != "All":
            filters.append({"difficulty": difficulty})

        where_clause = None
        if len(filters) == 1:
            where_clause = filters[0]
        elif len(filters) > 1:
            where_clause = {"$and": filters}

        try:
            results = self.collection.query(
                query_embeddings=[query_emb],
                n_results=n_results,
                where=where_clause
            )
        except Exception as e:
            # Fallback query without strict filters if too restrictive
            results = self.collection.query(
                query_embeddings=[query_emb],
                n_results=n_results
            )

        items = []
        if results and "documents" in results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0] if "metadatas" in results else [{}] * len(docs)
            distances = results["distances"][0] if "distances" in results else [0.0] * len(docs)
            for d, m, dist in zip(docs, metas, distances):
                items.append({
                    "document": d,
                    "metadata": m,
                    "distance": dist
                })
        return items

    def get_stats(self) -> Dict[str, Any]:
        """Return total indexed questions count and breakdown."""
        total = self.collection.count()
        return {"total_count": total}
