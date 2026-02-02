import os
import json
import subprocess
from typing import List, Dict, Any, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

DATA_DIR = "data"
DOCS_PATH = os.path.join(DATA_DIR, "docs.jsonl")
META_PATH = os.path.join(DATA_DIR, "meta.json")
INDEX_PATH = os.path.join(DATA_DIR, "index.faiss")

# ✅ Puna putanja do Ollama CLI (jer nije u PATH-u)
OLLAMA_EXE = r"C:\Users\miara\AppData\Local\Programs\Ollama\ollama.exe"

# ✅ Model koji si instalirao
OLLAMA_MODEL = "mistral"

# ✅ Prag relevantnosti (cosine sličnost na normaliziranim embeddingima)
# Ako je prenisko -> bolje pitati korisnika da precizira nego halucinirati
MIN_SCORE_THRESHOLD = 0.60


class RagStore:
    def __init__(self):
        if not (os.path.exists(DOCS_PATH) and os.path.exists(META_PATH) and os.path.exists(INDEX_PATH)):
            raise FileNotFoundError(
                "Nedostaje data/ indeks. Pokreni: python build_index.py "
                "(i provjeri da si u root folderu projekta)."
            )

        with open(META_PATH, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        self.embedder = SentenceTransformer(self.meta["model_name"])

        self.docs = []
        with open(DOCS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                self.docs.append(json.loads(line))

        self.index = faiss.read_index(INDEX_PATH)

    def search(self, query: str, k: int = 6) -> List[Dict[str, Any]]:
        q_emb = self.embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
        scores, idxs = self.index.search(q_emb, k)

        hits = []
        for score, idx in zip(scores[0].tolist(), idxs[0].tolist()):
            if idx < 0:
                continue
            d = self.docs[idx].copy()
            d["score"] = float(score)
            hits.append(d)
        return hits


def _context_from_hits(hits: List[Dict[str, Any]], max_chars: int = 8000) -> str:
    parts = []
    total = 0
    for i, h in enumerate(hits, start=1):
        block = f"[IZVOR {i}] URL: {h['url']}\n{h['text'].strip()}\n"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n".join(parts)


def _ollama_generate(prompt: str, timeout_sec: int = 180) -> str:
    """
    Poziva lokalni LLM kroz Ollama CLI i vraća odgovor.
    ✅ Windows-safe: input/output se šalje kao UTF-8 bytes da se izbjegne UnicodeEncodeError (charmap).
    """
    # Zamijeni “problematične” crtice (čistoća teksta)
    prompt = (
        prompt.replace("\u2010", "-")
              .replace("\u2011", "-")
              .replace("\u2013", "-")
              .replace("\u2014", "-")
    )

    prompt_bytes = prompt.encode("utf-8", errors="replace")

    try:
        result = subprocess.run(
            [OLLAMA_EXE, "run", OLLAMA_MODEL],
            input=prompt_bytes,
            capture_output=True,
            timeout=timeout_sec,
            check=False
        )
    except FileNotFoundError:
        raise RuntimeError(
            "Ollama CLI nije pronađena. Provjeri putanju do ollama.exe "
            f"(trenutno: {OLLAMA_EXE})"
        )

    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(stderr or "Ollama error")

    return result.stdout.decode("utf-8", errors="replace").strip()


def answer_llm_grounded(user_question: str, hits: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    """
    RAG + LLM:
    - dohvaća top hitove iz FAISS-a
    - LLM generira odgovor samo iz izvora
    - ako su hitovi slabi, traži preciziranje (threshold)
    """
    if not hits:
        return (
            "Ne mogu pronaći relevantne informacije u javno dostupnom sadržaju Općine Punat. "
            "Molim provjerite službenu web stranicu ili postavite preciznije pitanje.",
            []
        )

    # ✅ Threshold zaštita
    best = hits[0].get("score", 0.0)
    if best < MIN_SCORE_THRESHOLD:
        return (
            "Možeš malo preciznije? Npr. 'radno vrijeme općine', 'kontakt', 'obrasci', 'natječaji' ili naziv dokumenta. "
            "Trenutno nemam dovoljno relevantan izvor za siguran odgovor.",
            []
        )

    context = _context_from_hits(hits)

    prompt = f"""
Ti si AI asistent za web stranicu Općine Punat.
Odgovaraj isključivo na hrvatskom jeziku.
Koristi ISKLJUČIVO informacije iz navedenih izvora (IZVOR 1..N).
Ako informacija nije u izvorima, napiši: "Nije dostupno u javno dostupnim podacima koje imam."
Ne izmišljaj i ne pretpostavljaj.
Odgovor neka bude kratak, jasan i koristan (do 120 riječi).

Korisničko pitanje:
{user_question}

Javno dostupni izvori:
{context}
""".strip()

    answer_text = _ollama_generate(prompt, timeout_sec=180)

    # UI sources (dedup)
    sources = []
    seen = set()
    for h in hits[:6]:
        if h["url"] not in seen:
            seen.add(h["url"])
            sources.append({"url": h["url"], "score": h["score"]})

    return answer_text, sources
