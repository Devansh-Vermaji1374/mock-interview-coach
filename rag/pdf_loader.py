import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import pypdf

class PDFDocumentProcessor:
    """Extracts, parses, and structures interview questions and answers from PDFs."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract text page by page with metadata."""
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found at {pdf_path}")

        reader = pypdf.PdfReader(str(path))
        pages_content = []
        for idx, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            pages_content.append({
                "page_number": idx + 1,
                "text": text,
                "source": path.name
            })
        return pages_content

    def extract_structured_qa(self, pdf_path: str, default_company: str = "General Tech", default_role: str = "Software Development Engineer (SDE / SWE)") -> List[Dict[str, Any]]:
        """Extract individual interview questions and answers with metadata."""
        pages = self.extract_text_from_pdf(pdf_path)
        full_text = "\n".join([p["text"] for p in pages])
        
        # Regex pattern matching Q1., Q2., QUESTION 1:, etc.
        pattern = re.compile(
            r'(?:Q(?:UESTION)?\s*(\d+)[\.\:\s]+)(.*?)(?=(?:Q(?:UESTION)?\s*\d+[\.\:\s]+)|\Z)',
            re.DOTALL | re.IGNORECASE
        )
        matches = list(pattern.finditer(full_text))
        
        results = []
        if matches:
            for m in matches:
                q_num = m.group(1)
                body = m.group(2).strip()
                lines = [l.strip() for l in body.split("\n") if l.strip()]
                if not lines:
                    continue
                
                # Question is usually the first 1-2 lines before 'Answer' or paragraph
                question_text = lines[0]
                if len(lines) > 1 and not lines[1].lower().startswith("answer") and len(lines[0]) < 80:
                    question_text += " " + lines[1]
                
                # Categorize as HR vs Technical based on keywords
                lower_q = question_text.lower()
                is_hr = any(k in lower_q for k in [
                    "introduce yourself", "why should we hire", "why do you want", "conflict", 
                    "weakness", "strength", "disagree", "deadline", "team", "tell me about yourself", "leadership"
                ])
                interview_type = "HR / Behavioral Interview" if is_hr else "Technical Interview"
                
                # Estimate difficulty
                difficulty = "Senior / Lead" if any(k in lower_q for k in ["architecture", "distribute", "scale", "trade-off", "design"]) else "Junior / Entry Level" if any(k in lower_q for k in ["what is", "introduce", "basic", "explain oop"]) else "Mid Level"

                results.append({
                    "id": f"pdf_{Path(pdf_path).stem}_q{q_num}",
                    "question": question_text,
                    "content": body,
                    "company": default_company,
                    "role": default_role,
                    "type": interview_type,
                    "difficulty": difficulty,
                    "source": Path(pdf_path).name,
                    "verified": True
                })
        else:
            # Fallback chunking if no explicit Q&A numbering detected
            chunks = self._chunk_text(full_text)
            for idx, ch in enumerate(chunks):
                results.append({
                    "id": f"pdf_{Path(pdf_path).stem}_chunk_{idx}",
                    "question": f"Question chunk {idx+1}",
                    "content": ch,
                    "company": default_company,
                    "role": default_role,
                    "type": "Technical Interview",
                    "difficulty": "Mid Level",
                    "source": Path(pdf_path).name,
                    "verified": False
                })
        return results

    def _chunk_text(self, text: str) -> List[str]:
        words = text.split()
        chunks = []
        i = 0
        while i < len(words):
            chunk = " ".join(words[i:i + self.chunk_size])
            chunks.append(chunk)
            i += (self.chunk_size - self.chunk_overlap)
        return chunks
