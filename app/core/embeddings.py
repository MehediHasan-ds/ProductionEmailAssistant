"""Local Jina embeddings via ONNX Runtime.

Loads the quantized EuroBERT ONNX model directly and applies last-token pooling
and L2 normalization, matching the model's sentence-transformers pipeline.
Offline, CPU-only, provider-agnostic.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import onnxruntime as ort
import structlog
from transformers import AutoTokenizer

from app.config import Settings, get_settings

log = structlog.get_logger(__name__)


class JinaEmbedder:
    # TODO: verify BOS token is being prepended automatically
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._tokenizer = None
        self._session: ort.InferenceSession | None = None
        self._input_names: list[str] = []

    def _load(self) -> None:
        model_dir = Path(self._settings.embedder_model_dir)
        onnx_path = model_dir / "onnx" / self._settings.embedder_onnx_file
        self._tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
        self._session = ort.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"],
        )
        self._input_names = [i.name for i in self._session.get_inputs()]
        log.info(
            "embedder.loaded",
            onnx=str(onnx_path),
            inputs=self._input_names,
        )

    def _ensure_loaded(self) -> None:
        if self._session is None:
            self._load()

    def encode(self, texts: list[str]) -> np.ndarray:
        self._ensure_loaded()
        assert self._tokenizer is not None and self._session is not None

        prefixed = [self._settings.embedder_prefix + t for t in texts]
        enc = self._tokenizer(
            prefixed,
            padding=True,
            truncation=True,
            max_length=self._settings.embedder_max_length,
            return_tensors="np",
        )
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        # AutoTokenizer does not prepend BOS for this generic tokenizer class,
        # and EuroBERT requires <|begin_of_text|> at position 0 for correct RoPE.
        bos = np.full(
            (input_ids.shape[0], 1),
            self._tokenizer.bos_token_id,
            dtype=input_ids.dtype,
        )
        ones = np.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype)
        input_ids = np.concatenate([bos, input_ids], axis=1)
        attention_mask = np.concatenate([ones, attention_mask], axis=1)

        feeds: dict[str, np.ndarray] = {}
        for name in self._input_names:
            if name == "input_ids":
                feeds[name] = input_ids
            elif name == "attention_mask":
                feeds[name] = attention_mask
            elif name in enc:
                feeds[name] = enc[name]
        last_hidden = self._session.run(None, feeds)[0]

        seq_len = attention_mask.sum(axis=1) - 1
        rows = np.arange(last_hidden.shape[0])
        pooled = last_hidden[rows, seq_len].astype(np.float32)

        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return pooled / norms

    def cosine(self, a: str, b: str) -> float:
        vecs = self.encode([a, b])
        return float(vecs[0] @ vecs[1])

    def similarity_matrix(self, preds: list[str], refs: list[str]) -> np.ndarray:
        vecs = self.encode(list(preds) + list(refs))
        return vecs[: len(preds)] @ vecs[len(preds) :].T


@lru_cache(maxsize=1)
def get_embedder() -> JinaEmbedder:
    return JinaEmbedder(get_settings())
