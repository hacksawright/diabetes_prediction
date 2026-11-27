"""Local multilingual embedding and reranking on CPU."""

from functools import lru_cache
from pathlib import Path

import numpy as np

from ..config import ROOT

MODEL_DIR = ROOT / "artifacts" / "rag_models"
EMBEDDING_VERSION = "multilingual-e5-small-onnx-uint8-v1"
EMBEDDING_DIM = 384
MODEL_FILES = {
    "e5": {
        "tokenizer.json": "https://huggingface.co/Xenova/multilingual-e5-small/resolve/main/tokenizer.json",
        "model.onnx": "https://huggingface.co/Xenova/multilingual-e5-small/resolve/main/onnx/model_uint8.onnx",
    },
    "reranker": {
        "tokenizer.json": "https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1/resolve/main/tokenizer.json",
        "model.onnx": "https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1/resolve/main/onnx/model_quint8_avx2.onnx",
    },
}


@lru_cache(maxsize=2)
def _load_model(name: str):
    import onnxruntime as ort
    from tokenizers import Tokenizer

    directory = MODEL_DIR / name
    tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
    tokenizer.enable_truncation(max_length=512)
    tokenizer.enable_padding(pad_id=1, pad_token="<pad>")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    model = ort.InferenceSession(
        str(directory / "model.onnx"), sess_options=options, providers=["CPUExecutionProvider"]
    )
    return tokenizer, model


def available(name: str) -> bool:
    return all((MODEL_DIR / name / filename).is_file() for filename in MODEL_FILES[name])


def _run(name, texts):
    tokenizer, model = _load_model(name)
    output = []
    for offset in range(0, len(texts), 8):
        batch = tokenizer.encode_batch(texts[offset:offset + 8])
        masks = np.asarray([item.attention_mask for item in batch], dtype=np.int64)
        values = {
            "input_ids": np.asarray([item.ids for item in batch], dtype=np.int64),
            "attention_mask": masks,
            "token_type_ids": np.asarray([item.type_ids for item in batch], dtype=np.int64),
        }
        scores = model.run(None, {item.name: values[item.name] for item in model.get_inputs()})[0]
        if name == "e5":
            scores = (scores * masks[:, :, None]).sum(axis=1) / masks.sum(axis=1, keepdims=True)
            scores /= np.maximum(np.linalg.norm(scores, axis=1, keepdims=True), 1e-12)
        output.extend(scores.tolist())
    return output


def embed(texts: list[str], *, query: bool = False) -> list[list[float]]:
    prefix = "query: " if query else "passage: "
    return _run("e5", [prefix + text for text in texts])


def rerank(query: str, passages: list[str]) -> list[float]:
    return [row[0] for row in _run("reranker", [(query, passage) for passage in passages])]
