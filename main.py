from pathlib import Path

from pdf_parser import parse_pdf
from scripts.clean_markdown import main as run_cleaner

PROJECT_ROOT = Path(__file__).resolve().parent
PDF_PATH = PROJECT_ROOT / "data" / "Lesson-13.pdf"

if __name__ == "__main__":
    text = parse_pdf(
        PDF_PATH,
        PROJECT_ROOT / "output" / "output.md",
        PROJECT_ROOT / "output" / "output.txt",
    )
    print(text[:500])  # preview first 500 chars
    run_cleaner()