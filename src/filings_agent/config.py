from pathlib import Path

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    llm_model: str = "groq/openai/gpt-oss-20b"
    # A different model family grades answers, so the judge does not favour its own phrasing.
    judge_model: str = "groq/qwen/qwen3.8-27b"
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_collection: str = "filings"
    embed_model: str = "BAAI/bge-small-en-v1.5"
    data_dir: Path = ROOT / "data"
    max_agent_steps: int = 6
    llm_retries: int = 4  # retries after a 429 or connection error, waiting as the provider asks
    llm_rpm: int = 25  # under Groq's free-tier per-minute limit; 0 disables pacing


settings = Settings()


def load_companies() -> list[dict]:
    with open(ROOT / "config" / "companies.yaml") as f:
        return yaml.safe_load(f)["companies"]
