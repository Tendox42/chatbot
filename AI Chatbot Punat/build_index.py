import os
import re
import json
import time
import hashlib
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

BASE_URL = "https://www.punat.hr/"
OUT_DIR = "data"
DOCS_PATH = os.path.join(OUT_DIR, "docs.jsonl")
META_PATH = os.path.join(OUT_DIR, "meta.json")
INDEX_PATH = os.path.join(OUT_DIR, "index.faiss")

# Koliko URL-ova maksimalno crawlati (prototip)
MAX_PAGES = 150

# Chunking
CHUNK_SIZE_CHARS = 1200
CHUNK_OVERLAP_CHARS = 200

# Poštuj server
REQUEST_DELAY_SEC = 0.35
TIMEOUT_SEC = 15

USER_AGENT = "PunatTBIPrototypeBot/1.0 (educational prototype)"

# ✅ BITNO: preskoči binarne datoteke (PDF i sl.) – inače dobiješ "čudne znakove"
SKIP_EXTENSIONS = (
    ".pdf", ".doc", ".docx", ".xls", ".xlsx",
    ".ppt", ".pptx", ".zip", ".rar"
)


def is_same_domain(url: str, base: str) -> bool:
    return urlparse(url).netloc == urlparse(base).netloc


def normalize_url(url: str) -> str:
    url = url.split("#")[0]
    url = re.sub(r"/+$", "/", url)
    return url


def extract_visible_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    # grubo uklanjanje navigacije/header/footer (nije savršeno, ali ok za prototip)
    for sel in ["header", "footer", "nav", ".menu", ".navigation", ".breadcrumbs"]:
        for el in soup.select(sel):
            el.decompose()

    text = soup.get_text(separator="\n")
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines)


def split_into_chunks(text: str, size: int, overlap: int):
    if len(text) <= size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks


def stable_id(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def crawl_site(base_url: str, max_pages: int):
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    visited = set()
    queue = [base_url]
    pages = []

    pbar = tqdm(total=max_pages, desc="Crawling pages")

    while queue and len(visited) < max_pages:
        url = normalize_url(queue.pop(0))

        # ✅ preskoči PDF/binarne resurse
        if url.lower().endswith(SKIP_EXTENSIONS):
            visited.add(url)
            pbar.update(1)
            continue

        if url in visited:
            continue
        if not is_same_domain(url, base_url):
            continue

        try:
            r = session.get(url, timeout=TIMEOUT_SEC)
            if r.status_code != 200:
                visited.add(url)
                pbar.update(1)
                continue

            # ✅ dodatna zaštita: ako server vrati PDF content-type, preskoči
            ctype = (r.headers.get("content-type") or "").lower()
            if "application/pdf" in ctype:
                visited.add(url)
                pbar.update(1)
                continue

            html = r.text
        except Exception:
            visited.add(url)
            pbar.update(1)
            continue

        visited.add(url)

        soup = BeautifulSoup(html, "lxml")
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if not href:
                continue
            abs_url = normalize_url(urljoin(url, href))

            # preskoči PDF/binarno već u fazi stavljanja u queue
            if abs_url.lower().endswith(SKIP_EXTENSIONS):
                continue

            if is_same_domain(abs_url, base_url) and abs_url not in visited:
                queue.append(abs_url)

        text = extract_visible_text(html)
        if len(text) > 300:
            pages.append({"url": url, "text": text})

        pbar.update(1)
        time.sleep(REQUEST_DELAY_SEC)

    pbar.close()
    return pages


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    pages = crawl_site(BASE_URL, MAX_PAGES)
    print(f"Fetched pages: {len(pages)}")

    docs = []
    for p in pages:
        url = p["url"]
        chunks = split_into_chunks(p["text"], CHUNK_SIZE_CHARS, CHUNK_OVERLAP_CHARS)
        for i, ch in enumerate(chunks):
            docs.append({
                "id": stable_id(f"{url}::chunk::{i}::{ch[:80]}"),
                "url": url,
                "chunk_index": i,
                "text": ch,
            })

    # dedup po tekstu
    unique = {}
    for d in docs:
        h = stable_id(d["text"])
        if h not in unique:
            unique[h] = d
    docs = list(unique.values())
    print(f"Chunks (deduped): {len(docs)}")

    with open(DOCS_PATH, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    model = SentenceTransformer(model_name)

    texts = [d["text"] for d in docs]
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine via normalized vectors
    index.add(embeddings.astype(np.float32))
    faiss.write_index(index, INDEX_PATH)

    meta = {
        "base_url": BASE_URL,
        "model_name": model_name,
        "count": len(docs),
        "embedding_dim": int(dim),
        "index_type": "IndexFlatIP (cosine via normalized embeddings)",
        "created_at_unix": int(time.time()),
        "skipped_extensions": list(SKIP_EXTENSIONS),
        "max_pages": MAX_PAGES,
        "chunk_size_chars": CHUNK_SIZE_CHARS,
        "chunk_overlap_chars": CHUNK_OVERLAP_CHARS,
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("Saved:")
    print(f"- {DOCS_PATH}")
    print(f"- {INDEX_PATH}")
    print(f"- {META_PATH}")


if __name__ == "__main__":
    main()
