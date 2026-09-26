from pathlib import Path

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    llm_model: str = "groq/openai/gpt-oss-20b"
    # A larger model grades answers against the hand-checked expected ones.
    judge_model: str = "groq/openai/gpt-oss-120b"
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_collection: str = "filings"
    embed_model: str = "BAAI/bge-small-en-v1.5"
    data_dir: Path = ROOT / "data"
    max_agent_steps: int = 5
    # Groq's free tier refuses any request over 8K tokens (input plus max output), so the
    # agent's context is trimmed to this many characters (~4K tokens); 0 disables trimming.
    max_context_chars: int = 14000
    max_answer_tokens: int = 2048
    llm_retries: int = 4  # retries after a 429 or connection error, waiting as the provider asks
    llm_rpm: int = 25  # under Groq's free-tier per-minute limit; 0 disables pacing


settings = Settings()


def load_companies() -> list[dict]:
    with open(ROOT / "config" / "companies.yaml") as f:
        return yaml.safe_load(f)["companies"]


def report_url(doc_id: str, page: int | None = None) -> str | None:
    """Public URL of a filing, opened at a page when given (PDF viewers honour #page=)."""
    company, fy, _ = doc_id.split("_", 2)
    for c in load_companies():
        if c["id"] == company and c["annual_reports"].get(fy):
            url = c["annual_reports"][fy]
            return f"{url}#page={page}" if page else url
    return None
