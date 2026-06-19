"""Milestone 1 embedder safety checks: BOS handling, normalization, and cosine sanity."""
import numpy as np

from app.core.embeddings import get_embedder

BOS_ID = 128000


def main() -> None:
    embedder = get_embedder()

    vecs = embedder.encode(["a short test sentence"])
    assert embedder._tokenizer is not None
    bos = embedder._tokenizer.bos_token_id
    print("tokenizer bos_token_id:", bos)
    assert bos == BOS_ID, f"expected BOS {BOS_ID}, got {bos}"

    print("embedding shape:", vecs.shape)
    assert vecs.shape == (1, 768), vecs.shape
    norm = float(np.linalg.norm(vecs[0]))
    print("L2 norm:", round(norm, 4))
    assert abs(norm - 1.0) < 1e-3, "embeddings must be L2-normalized"

    related = embedder.cosine("urgent payment reminder", "overdue invoice notice")
    unrelated = embedder.cosine("urgent payment reminder", "a recipe for banana bread")
    print("cosine related:", round(related, 4))
    print("cosine unrelated:", round(unrelated, 4))
    assert related > unrelated, "related pair should score higher than unrelated"
    assert related < 0.99, "cosine near 1.0 suggests wrong pooling"
    print("cosine sanity ok")


if __name__ == "__main__":
    main()
