"""TF-IDF feature extraction for raw API-call-sequence text data.

Fits a TfidfVectorizer over n-grams of API call names and stores the
resulting sparse matrix on disk (.npz) so training scripts don't need
to re-fit it every run.
"""
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer


def fit_tfidf(sequences, ngram_range=(3, 3), smooth_idf=True, norm="l2",
              sublinear_tf=True, max_df=0.9) -> tuple:
    """Fit a TfidfVectorizer on a list of API-call sequence strings.

    Returns (sparse_matrix, fitted_vectorizer).
    """
    vectorizer = TfidfVectorizer(
        ngram_range=tuple(ngram_range),
        smooth_idf=smooth_idf,
        norm=norm,
        sublinear_tf=sublinear_tf,
        max_df=max_df,
    )
    embedding = vectorizer.fit_transform(sequences)
    return embedding, vectorizer


def save_embedding(embedding, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(path, embedding)


def load_embedding(path: str):
    if not Path(path).exists():
        raise FileNotFoundError(
            f"TF-IDF embedding not found at '{path}'. Run the preprocessing "
            "step first (see README) to generate it."
        )
    return sparse.load_npz(path)


def load_raw_sequences(text_path: str) -> list:
    """Load raw, comma-separated API-call sequences (one per line)."""
    with open(text_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines


def main():
    import argparse

    from src.utils.config import load_config

    parser = argparse.ArgumentParser(description="Build and cache TF-IDF features.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    sequences = load_raw_sequences(cfg["data"]["raw_text_path"])
    embedding, _ = fit_tfidf(sequences, **cfg["tfidf"])
    save_embedding(embedding, cfg["data"]["tfidf_embedding_path"])
    print(f"Saved TF-IDF embedding with shape {embedding.shape} "
          f"to {cfg['data']['tfidf_embedding_path']}")


if __name__ == "__main__":
    main()
