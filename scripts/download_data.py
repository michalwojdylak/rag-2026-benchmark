"""Download the unified A-RAG benchmark release (MuSiQue, HotpotQA, 2Wiki, GraphRAG-Bench Medical/Novel)."""

import urllib.request
from pathlib import Path

BASE = "https://huggingface.co/datasets/Ayanami0730/rag_test/resolve/main"
OUT = Path(__file__).resolve().parents[1] / "data" / "raw"

for ds in ["musique", "hotpotqa", "2wikimultihop", "medical", "novel"]:
    (OUT / ds).mkdir(parents=True, exist_ok=True)
    for name in ["chunks.json", "questions.json"]:
        target = OUT / ds / name
        if not target.exists():
            print("downloading", ds, name)
            urllib.request.urlretrieve(f"{BASE}/{ds}/{name}", target)
