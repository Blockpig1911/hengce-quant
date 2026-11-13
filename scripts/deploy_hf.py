"""Deploy the app to Hugging Face Spaces.

Usage:
    set HF_TOKEN=hf_xxx        (Windows cmd)
    python scripts/deploy_hf.py
"""
from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]


def main():
    token = os.getenv("HF_TOKEN", "").strip()
    if not token:
        raise SystemExit("HF_TOKEN is not set")
    api = HfApi(token=token)
    user = api.whoami()["name"]
    repo_id = f"{user}/hengce-quant"
    api.create_repo(repo_id, repo_type="space", space_sdk="streamlit",
                    private=False, exist_ok=True)
    api.upload_folder(
        folder_path=str(ROOT),
        repo_id=repo_id,
        repo_type="space",
        ignore_patterns=[".git/**", "**/__pycache__/**", "**/*.pyc",
                         ".streamlit/secrets.toml", "cloudflared.exe"],
    )
    api.upload_file(
        path_or_fileobj=str(ROOT / "deploy" / "README_HF.md"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="space",
    )
    print(f"https://huggingface.co/spaces/{repo_id}")
    print(f"https://{user}-hengce-quant.hf.space")


if __name__ == "__main__":
    main()
