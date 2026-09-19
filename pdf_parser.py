import os
from pathlib import Path
from llama_cloud import LlamaCloud
from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env"))

api_key = os.getenv("LLAMA_CLOUD_API_KEY")
if not api_key:
    raise RuntimeError(
        "LLAMA_CLOUD_API_KEY is missing. Add it to the project .env file "
        "or set it in the shell environment."
    )

client = LlamaCloud(api_key=api_key)


def parse_pdf(file_path, output_md_path="output/output.md", output_txt_path="output/output.txt"):
    """Upload a PDF to LlamaCloud, parse it, and save markdown + text to disk."""
    file_obj = client.files.create(file=file_path, purpose="parse")

    result = client.parsing.parse(
        file_id=file_obj.id,
        tier="agentic",
        version="latest",
        expand=["markdown_full", "text_full"],
    )

    markdown_text = result.markdown_full or ""
    plain_text = result.text_full or ""

    markdown_path = Path(output_md_path)
    text_path = Path(output_txt_path)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(markdown_text, encoding="utf-8")
    text_path.write_text(plain_text, encoding="utf-8")

    print(f"Wrote {len(markdown_text)} chars of markdown, {len(plain_text)} chars of text")
    return plain_text