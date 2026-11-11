import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
CHROMA_PERSIST_DIRECTORY = str(BASE_DIR / os.getenv("CHROMA_PERSIST_DIRECTORY", "database/chroma"))
DATA_DIR = BASE_DIR / "data"
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gemini-3.5-flash-lite")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

TARGET_PDF_PATH = DATA_DIR / "30-SOFTWARE-ENGINEERING-Interview-Questions-and-Answers_FINAL.pdf"

SUPPORTED_COMPANIES = ["Google", "Amazon", "Microsoft", "Infosys", "TCS", "General Tech"]
INTERVIEW_ROLES = [
    "Software Development Engineer (SDE / SWE)",
    "Frontend Developer",
    "Backend Developer",
    "Full-Stack Engineer",
    "DevOps / Cloud Engineer",
    "Data Engineer / AI Engineer"
]
DIFFICULTY_LEVELS = ["Junior / Entry Level", "Mid Level", "Senior / Lead"]
INTERVIEW_TYPES = ["Technical Interview", "HR / Behavioral Interview"]


