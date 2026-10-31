"""Pretrained model loading, feature adapters, and embedding identity."""

import json
from copy import deepcopy
from hashlib import sha256
from importlib.metadata import version
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .images import ImageSource, file_sha256, load_rgb

DEFAULT_MODEL = "google/siglip2-base-patch16-224"
QUALITY_MODEL = "google/siglip2-so400m-patch14-384"
PIPELINE_VERSION = 1


def identity_fingerprint(identity: dict) -> str:
    canonical = json.dumps(identity, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return sha256(canonical.encode("utf-8")).hexdigest()


def pooled_features(output: Any) -> torch.Tensor:
    """Support tensor-returning and pooled ModelOutput feature APIs."""
    if isinstance(output, torch.Tensor):
        return output
    pooled = getattr(output, "pooler_output", None)
    if not isinstance(pooled, torch.Tensor):
        raise ValueError("Model did not return a pooled embedding tensor")
    return pooled


def feature_matrix(features: torch.Tensor) -> np.ndarray:
    if features.ndim != 2 or min(features.shape) == 0:
        raise ValueError("Global embeddings must be a nonempty two-dimensional matrix")
    features = features.detach().float().cpu()
    if not torch.isfinite(features).all():
        raise ValueError("Embeddings must contain only finite values")
    norms = torch.linalg.vector_norm(features, dim=1, keepdim=True)
    if not torch.isfinite(norms).all():
        raise ValueError("Embedding norms must be finite")
    if (norms == 0).any():
        raise ValueError("Zero embeddings cannot be normalized")
    return np.ascontiguousarray((features / norms).numpy(), dtype=np.float32)


def _device(requested: str) -> torch.device:
    if requested == "auto":
        requested = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    if device.type not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"Unsupported device: {requested}")
    return device


def _batches(items: list, batch_size: int):
    if isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if not items:
        raise ValueError("Provide at least one input")
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


class SigLIPEmbedder:
    """Load a pretrained checkpoint once; return normalized NumPy embeddings."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        revision: str | None = None,
        device: str = "auto",
        local_files_only: bool = False,
    ):
        from huggingface_hub import hf_hub_download
        from transformers import AutoConfig, AutoModel, AutoProcessor

        self.device = _device(device)
        local_path = Path(model_id).expanduser()
        if local_path.is_dir():
            source = str(local_path.resolve())
            weights = sorted(local_path.glob("*.safetensors"))
            if not weights:
                raise ValueError("Local checkpoint must contain safetensors weights")
            resolved_revision = "local:" + identity_fingerprint(
                {p.name: file_sha256(p) for p in weights}
            )
            if revision is not None and revision != resolved_revision:
                raise ValueError("Local checkpoint weights changed; rebuild the index")
            config_path = local_path / "config.json"
            revision_args = {}
        else:
            if model_id.startswith(("/", "./", "../", "~")):
                raise ValueError(f"Local checkpoint directory does not exist: {model_id}")
            source = model_id
            config_path = Path(
                hf_hub_download(
                    model_id, "config.json", revision=revision, local_files_only=local_files_only
                )
            )
            resolved_revision = config_path.parent.name
            revision_args = {"revision": resolved_revision}

        common = dict(local_files_only=local_files_only, trust_remote_code=False, **revision_args)
        config = AutoConfig.from_pretrained(source, **common)
        if config.model_type not in {"siglip", "siglip2"}:
            raise ValueError(f"Expected a SigLIP checkpoint, got model_type={config.model_type}")
        self.processor = AutoProcessor.from_pretrained(source, **common)
        self.model = (
            AutoModel.from_pretrained(
                source, config=config, use_safetensors=True, dtype=torch.float32, **common
            )
            .to(self.device)
            .eval()
        )
        self.max_text_length = int(config.text_config.max_position_embeddings)

        tokenizer = self.processor.tokenizer
        backend = getattr(tokenizer, "backend_tokenizer", None)
        tokenizer_recipe = (
            backend.to_str()
            if backend is not None
            else json.dumps(
                {"vocab": tokenizer.get_vocab(), "kwargs": tokenizer.init_kwargs},
                sort_keys=True,
                default=str,
            )
        )
        processing = {
            "image_processor": self.processor.image_processor.to_dict(),
            "tokenizer": sha256(tokenizer_recipe.encode()).hexdigest(),
            "special_tokens": tokenizer.special_tokens_map,
            "max_text_length": self.max_text_length,
            "padding": "max_length",
            "text_template": "{text}",
            "rgb": "exif_transpose_alpha_on_white",
        }
        processing_hash = sha256(
            json.dumps(processing, sort_keys=True, ensure_ascii=False, default=str).encode()
        ).hexdigest()
        self._identity = {
            "model_id": source,
            "revision": resolved_revision,
            "config_sha256": file_sha256(config_path),
            "processor_sha256": processing_hash,
            "pipeline_version": PIPELINE_VERSION,
            "pooling": "checkpoint_pooled_output",
            "normalization": "l2_float32",
            "text_max_length": self.max_text_length,
            "runtime_versions": {
                name: version(name) for name in ("transformers", "torch", "Pillow")
            },
        }

    @property
    def identity(self) -> dict:
        return deepcopy(self._identity)

    def encode_images(self, images: list[ImageSource], batch_size: int = 8) -> np.ndarray:
        matrices = []
        with torch.inference_mode():
            for batch in _batches(list(images), batch_size):
                rgb_images = [load_rgb(image) for image in batch]
                inputs = self.processor(images=rgb_images, return_tensors="pt").to(self.device)
                features = feature_matrix(pooled_features(self.model.get_image_features(**inputs)))
                if features.shape[0] != len(batch):
                    raise ValueError("Image encoder returned the wrong number of rows")
                matrices.append(features)
        return np.concatenate(matrices, axis=0)

    def encode_texts(self, texts: str | list[str], batch_size: int = 8) -> np.ndarray:
        matrices = []
        texts = [texts] if isinstance(texts, str) else list(texts)
        with torch.inference_mode():
            for batch in _batches(texts, batch_size):
                if any(not isinstance(text, str) or not text.strip() for text in batch):
                    raise ValueError("Texts must be nonempty strings")
                tokens = self.processor.tokenizer(batch, padding=False, truncation=False)
                if any(len(ids) > self.max_text_length for ids in tokens["input_ids"]):
                    raise ValueError(
                        f"Text exceeds {self.max_text_length} tokens; shorten the query"
                    )
                inputs = self.processor(
                    text=batch,
                    padding="max_length",
                    max_length=self.max_text_length,
                    truncation=True,
                    return_tensors="pt",
                ).to(self.device)
                features = feature_matrix(pooled_features(self.model.get_text_features(**inputs)))
                if features.shape[0] != len(batch):
                    raise ValueError("Text encoder returned the wrong number of rows")
                matrices.append(features)
        return np.concatenate(matrices, axis=0)
