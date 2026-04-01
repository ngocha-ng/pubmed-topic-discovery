import re
import csv
import pickle
import numpy as np
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz
from sentence_transformers import SentenceTransformer

# ── config ────────────────────────────────────────────────────────────────────

RAW_DIR = Path("data/raw")

DOMAIN_FILES = {
    "drug_discovery": RAW_DIR / "drug_discovery.csv",
    "oncology":       RAW_DIR / "oncology.csv",
}

# Choose your model:
# "allenai/scibert_scivocab_uncased"   — stronger, heavier (~440MB)
# "pritamdeka/S-PubMedBert-MS-MARCO"  — lighter, faster (~420MB)
EMBEDDING_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"

BATCH_SIZE   = 64    # lower to 32 if you run out of memory
OUTPUT_DIR   = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── text cleaning ─────────────────────────────────────────────────────────────

def clean_for_tfidf(text: str) -> str:
    """Aggressive cleaning for bag-of-words: lowercase, no HTML, no punctuation."""
    text = re.sub(r"<[^>]+>", " ", text)           # strip HTML tags
    text = re.sub(r"[\u2028\u2029\r\n\t]", " ", text)  # unusual line terminators
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)       # strip punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_for_embedding(text: str) -> str:
    """Light cleaning for embeddings — models handle tokenization themselves."""
    text = re.sub(r"<[^>]+>", " ", text)           # strip HTML tags
    text = re.sub(r"[\u2028\u2029]", " ", text)    # unusual line terminators
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── load CSV ──────────────────────────────────────────────────────────────────

def load_csv(filepath: str) -> list:
    rows = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


# ── TF-IDF ────────────────────────────────────────────────────────────────────

def build_tfidf(texts_clean: list, domain: str):
    print(f"  Building TF-IDF matrix for {len(texts_clean):,} documents...")

    vectorizer = TfidfVectorizer(
        max_features=50_000,
        ngram_range=(1, 2),     # unigrams + bigrams
        min_df=3,               # ignore very rare terms
        max_df=0.95,            # ignore near-universal terms
        sublinear_tf=True,      # apply log normalization
    )
    matrix = vectorizer.fit_transform(texts_clean)
    print(f"  TF-IDF shape: {matrix.shape}")

    # Save sparse matrix and vectorizer
    save_npz(OUTPUT_DIR / f"{domain}_tfidf.npz", matrix)
    with open(OUTPUT_DIR / f"{domain}_tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    print(f"  Saved → {domain}_tfidf.npz + {domain}_tfidf_vectorizer.pkl")
    return matrix, vectorizer


# ── embeddings ────────────────────────────────────────────────────────────────

def build_embeddings(texts_raw: list, domain: str, model: SentenceTransformer):
    print(f"  Generating embeddings for {len(texts_raw):,} abstracts...")
    print(f"  Batch size: {BATCH_SIZE} — this will take a few minutes...")

    embeddings = model.encode(
        texts_raw,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # unit vectors — good for cosine similarity
    )

    out_path = OUTPUT_DIR / f"{domain}_embeddings.npy"
    np.save(out_path, embeddings)
    print(f"  Saved → {domain}_embeddings.npy  shape: {embeddings.shape}")
    return embeddings


# ── save metadata ─────────────────────────────────────────────────────────────

def save_metadata(rows: list, domain: str):
    """Save a cleaned metadata CSV aligned with the embedding rows."""
    out_path = OUTPUT_DIR / f"{domain}_metadata.csv"
    fields = ["pmid", "title", "pub_date", "keywords", "mesh_terms", "domain"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})
    print(f"  Saved → {domain}_metadata.csv  ({len(rows):,} rows)")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    print("  (downloading on first run — ~420MB)\n")
    model = SentenceTransformer(EMBEDDING_MODEL)

    for domain, filepath in DOMAIN_FILES.items():
        if not Path(filepath).exists():
            print(f"Skipping {domain} — file not found: {filepath}")
            continue

        print("=" * 52)
        print(f"  Domain: {domain}")
        print("=" * 52)

        rows = load_csv(filepath)
        print(f"  Loaded {len(rows):,} records\n")

        # Drop rows with no abstract
        rows = [r for r in rows if r.get("abstract", "").strip()]
        print(f"  {len(rows):,} records have abstracts\n")

        abstracts_raw   = [clean_for_embedding(r["abstract"]) for r in rows]
        abstracts_clean = [clean_for_tfidf(r["abstract"]) for r in rows]

        build_tfidf(abstracts_clean, domain)
        print()
        build_embeddings(abstracts_raw, domain, model)
        print()
        save_metadata(rows, domain)
        print()

    print("Phase 2 complete. Output files in ./data/processed/")
    print()
    print("  Per domain:")
    print("    {domain}_tfidf.npz              — sparse TF-IDF matrix")
    print("    {domain}_tfidf_vectorizer.pkl   — fitted vectorizer (for top keywords)")
    print("    {domain}_embeddings.npy         — sentence embeddings (float32)")
    print("    {domain}_metadata.csv           — aligned metadata (pmid, title, etc.)")


if __name__ == "__main__":
    main()
