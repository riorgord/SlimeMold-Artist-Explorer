"""
本地 Anima 编码器 — 使用 ComfyUI 原生 Qwen3_06B 实现，与 Anima 向量库对齐。
加载 Qwen3 0.6B checkpoint，本地编码文本为 1024 维向量。
"""
import numpy as np
import torch
from pathlib import Path


class AnimaEncoder:
    def __init__(self, ckpt_path: str, device: str = None):
        self.ckpt_path = Path(ckpt_path)
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._tokenizer = None
        self._loaded = False

    def load(self) -> bool:
        if self._loaded:
            return True
        try:
            import safetensors.torch
        except ImportError:
            return False

        state = safetensors.torch.load_file(str(self.ckpt_path))

        # Auto-detect layer count
        layer_indices = set()
        for k in state:
            parts = k.split(".")
            for i, p in enumerate(parts):
                if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                    layer_indices.add(int(parts[i + 1]))
        n_layers = max(layer_indices) + 1 if layer_indices else 28

        # Remove lm_head from state dict (not needed for encoding)
        state = {k: v for k, v in state.items() if not k.startswith("lm_head")}

        from engines.comfy_clip.anima import Qwen3_06BModel, AnimaTokenizer

        dtype = torch.float32
        device = self._device

        model_options = {
            "model_name": "qwen3_06b",
            "qwen3_06b_model_config": {"num_hidden_layers": n_layers},
        }

        self._model = Qwen3_06BModel(
            device=device, dtype=dtype,
            attention_mask=True,
            model_options=model_options
        )
        self._model.to(device)

        missing, unexpected = self._model.load_sd(state)
        if missing:
            m = [k for k in missing if "lm_head" not in k]
            if m:
                print(f"[Anima Qwen3] 缺失键 (前5): {m[:5]}")
        if unexpected:
            print(f"[Anima Qwen3] 多余键 (前5): {list(unexpected)[:5]}")

        self._model.eval()
        self._tokenizer = AnimaTokenizer()
        self._loaded = True
        return True

    def encode(self, text: str) -> np.ndarray:
        if not self._loaded:
            raise RuntimeError("Anima 编码器未加载")

        # Anima uses bare tokenization — no chat template (verified against ComfyUI source)
        tw = self._tokenizer.tokenize_with_weights(text)["qwen3_06b"]

        with torch.no_grad():
            result = self._model.encode_token_weights(tw)
        out = result[0]

        # Mean pool across sequence → 1024-dim (matches the Anima vector library build)
        vec = out[0].mean(dim=0).cpu().numpy().astype(np.float32)

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec


_LOCAL_ENCODER = None


def get_encoder(ckpt_path: str = None):
    global _LOCAL_ENCODER
    if ckpt_path is None:
        return None
    ckpt_path = str(ckpt_path)
    if _LOCAL_ENCODER is not None and str(_LOCAL_ENCODER.ckpt_path) == ckpt_path:
        return _LOCAL_ENCODER
    _LOCAL_ENCODER = AnimaEncoder(ckpt_path)
    return _LOCAL_ENCODER


def encode_text_local(text: str, ckpt_path: str) -> np.ndarray:
    encoder = get_encoder(ckpt_path)
    if encoder is None:
        raise RuntimeError("未指定 Anima 编码器路径")
    if not encoder._loaded:
        encoder.load()
    return encoder.encode(text)