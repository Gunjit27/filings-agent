"""Create or update the Hugging Face Space and copy runtime secrets into it.

Usage: HF_TOKEN=... python deploy/push_space.py <user>/<space>
"""

import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
FILES = ["Dockerfile", "pyproject.toml", "README.md", "src", "config", "app"]
SECRETS = ["GROQ_API_KEY", "QDRANT_URL", "QDRANT_API_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"]


def main(space: str) -> None:
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(space, repo_type="space", space_sdk="docker", exist_ok=True)
    for name in SECRETS:
        if os.environ.get(name):
            api.add_space_secret(space, name, os.environ[name])
    api.upload_folder(
        repo_id=space,
        repo_type="space",
        folder_path=ROOT,
        allow_patterns=[f"{f}/**" if (ROOT / f).is_dir() else f for f in FILES],
        ignore_patterns=["**/__pycache__/**"],
        commit_message="Deploy from GitHub",
    )
    # The Space README carries the Spaces config header; it replaces the repo README there.
    api.upload_file(
        repo_id=space,
        repo_type="space",
        path_or_fileobj=ROOT / "deploy" / "space_README.md",
        path_in_repo="README.md",
        commit_message="Space card",
    )
    print(f"https://huggingface.co/spaces/{space}")


if __name__ == "__main__":
    main(sys.argv[1])
