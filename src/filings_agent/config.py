from pathlib import Path

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    llm_model: str = "gemini/gemini-2.5-flash"
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_collection: str = "filings"
    embed_model: str = "BAAI/bge-small-en-v1.5"
    data_dir: Path = ROOT / "data"
    max_agent_steps: int = 6


settings = Settings()


def load_companies() -> list[dict]:
    with open(ROOT / "config" / "companies.yaml") as f:
        return yaml.safe_load(f)["companies"]
