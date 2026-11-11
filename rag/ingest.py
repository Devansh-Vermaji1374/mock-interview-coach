import json
from pathlib import Path
from rag.pdf_loader import PDFDocumentProcessor
from rag.vector_store import InterviewVectorStore
import config

def ingest_all_knowledge():
    print("--- Starting Knowledge Ingestion ---")
    vector_store = InterviewVectorStore(persist_directory=config.CHROMA_PERSIST_DIRECTORY)
    processor = PDFDocumentProcessor()

    # 1. Ingest PDF
    pdf_path = config.TARGET_PDF_PATH
    if pdf_path.exists():
        print(f"Loading questions from PDF: {pdf_path}")
        pdf_docs = processor.extract_structured_qa(
            str(pdf_path),
            default_company="General Tech",
            default_role="Software Development Engineer (SDE / SWE)"
        )
        print(f"Extracted {len(pdf_docs)} structured interview questions from PDF.")
        vector_store.add_documents(pdf_docs)
    else:
        print(f"Warning: PDF not found at {pdf_path}")

    # 2. Ingest Sample Company Dataset
    sample_file = config.DATA_DIR / "sample_company_data.json"
    if sample_file.exists():
        print(f"Loading sample company questions from: {sample_file}")
        with open(sample_file, "r", encoding="utf-8-sig") as f:
            company_data = json.load(f)
            
        formatted_docs = []
        for i, item in enumerate(company_data):
            formatted_docs.append({
                "id": f"comp_{item['company'].lower()}_{i}",
                "question": item["question"],
                "content": f"Expected Guidelines: {item['guidelines']}\nSample Historical Feedback: {item.get('sample_feedback', '')}",
                "company": item["company"],
                "role": item["role"],
                "type": item["type"],
                "difficulty": item["difficulty"],
                "source": "Curated Company Benchmark (Synthetic)",
                "verified": False
            })
        print(f"Prepared {len(formatted_docs)} company-specific benchmark questions.")
        vector_store.add_documents(formatted_docs)

    stats = vector_store.get_stats()
    print(f"--- Ingestion Complete! Total indexed items in ChromaDB: {stats['total_count']} ---")

if __name__ == "__main__":
    ingest_all_knowledge()

