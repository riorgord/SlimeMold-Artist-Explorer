"""
本地 SDXL CLIP 编码器 — 使用 ComfyUI 原生 CLIP 模型实现，与向量库 100% 对齐。
从 SDXL checkpoint 提取 CLIP 权重，本地编码文本。不依赖 ComfyUI 运行时。
"""
import numpy as np
import torch
from pathlib import Path


def _short_hash(path: Path) -> str:
    size = path.stat().st_size
    return f"{path.stem}_{size}"


class LocalSDXLEncoder:
    def __init__(self, ckpt_path: str, cache_dir: str = "./data/cached_clip", device: str = None):
        self.ckpt_path = Path(ckpt_path)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None      # SDXLClipModel
        self._tokenizer = None  # SDXLTokenizer
        self._loaded = False

    def _load_or_extract(self) -> dict:
        """加载 CLIP 权重。首次从完整 checkpoint 全量加载提取并缓存，之后走缓存。"""
        import safetensors.torch
        cache_name = f"{_short_hash(self.ckpt_path)}_clip.safetensors"
        cache_path = self.cache_dir / cache_name

        if cache_path.exists():
            return safetensors.torch.load_file(str(cache_path))

        print("🔧 首次加载: 从 checkpoint 提取 CLIP 权重...")
        full_state = safetensors.torch.load_file(str(self.ckpt_path))

        clip_state = {}
        for k, v in full_state.items():
            if k.startswith("conditioner.embedders.0.transformer."):
                clip_state[f"clip_l.{k[len('conditioner.embedders.0.transformer.'):]}"] = v
            elif k.startswith("conditioner.embedders.1.model."):
                clip_state[f"clip_g.{k[len('conditioner.embedders.1.model.'):]}"] = v
        del full_state

        if not clip_state:
            raise ValueError("未找到 conditioner.embedders 格式的 CLIP 权重")

        safetensors.torch.save_file(clip_state, str(cache_path))
        print(f"💾 CLIP 权重已缓存: {cache_path}  ({cache_path.stat().st_size // 1024 // 1024}MB)")
        return safetensors.torch.load_file(str(cache_path))

    @staticmethod
    def _split_in_proj(in_proj_weight, in_proj_bias=None):
        """将 OpenCLIP 的 in_proj [3*hidden, hidden] 拆为 q/k/v_proj [hidden, hidden]"""
        dim = in_proj_weight.shape[0] // 3
        q_w = in_proj_weight[:dim, :]
        k_w = in_proj_weight[dim:2*dim, :]
        v_w = in_proj_weight[2*dim:, :]
        result = {"q_proj.weight": q_w, "k_proj.weight": k_w, "v_proj.weight": v_w}
        if in_proj_bias is not None:
            result["q_proj.bias"] = in_proj_bias[:dim]
            result["k_proj.bias"] = in_proj_bias[dim:2*dim]
            result["v_proj.bias"] = in_proj_bias[2*dim:]
        return result

    @staticmethod
    def _convert_clip_g(clip_g_state: dict):
        """将 OpenCLIP 格式的 CLIP-G 权重转为 ComfyUI CLIPTextModel 格式。
        返回 (mapped_dict, detected_config_overrides)。"""
        mapped = {}
        if "text_projection" in clip_g_state:
            mapped["text_projection.weight"] = clip_g_state["text_projection"]

        prefix = "transformer.resblocks."
        layer_indices = set()
        for k in clip_g_state:
            if k.startswith(prefix):
                idx_str = k[len(prefix):].split(".")[0]
                if idx_str.isdigit():
                    layer_indices.add(int(idx_str))
        n_layers = max(layer_indices) + 1 if layer_indices else 24

        # 检测 hidden_size
        hidden_size = None
        for idx in sorted(layer_indices):
            w_key = f"transformer.resblocks.{idx}.attn.in_proj_weight"
            if w_key in clip_g_state:
                hidden_size = clip_g_state[w_key].shape[0] // 3
                break

        for layer_idx in sorted(layer_indices):
            base = f"transformer.resblocks.{layer_idx}."
            hf_base = f"text_model.encoder.layers.{layer_idx}."

            # in_proj → q, k, v
            w_key = base + "attn.in_proj_weight"
            b_key = base + "attn.in_proj_bias"
            if w_key in clip_g_state:
                w = clip_g_state[w_key]
                b = clip_g_state.get(b_key)
                parts = LocalSDXLEncoder._split_in_proj(w, b)
                for sub_k, sub_v in parts.items():
                    mapped[hf_base + "self_attn." + sub_k] = sub_v
                out_key = base + "attn.out_proj"
                if out_key + ".weight" in clip_g_state:
                    mapped[hf_base + "self_attn.out_proj.weight"] = clip_g_state[out_key + ".weight"]
                if out_key + ".bias" in clip_g_state:
                    mapped[hf_base + "self_attn.out_proj.bias"] = clip_g_state[out_key + ".bias"]

            # mlp
            if base + "mlp.c_fc.weight" in clip_g_state:
                mapped[hf_base + "mlp.fc1.weight"] = clip_g_state[base + "mlp.c_fc.weight"]
            if base + "mlp.c_fc.bias" in clip_g_state:
                mapped[hf_base + "mlp.fc1.bias"] = clip_g_state[base + "mlp.c_fc.bias"]
            if base + "mlp.c_proj.weight" in clip_g_state:
                mapped[hf_base + "mlp.fc2.weight"] = clip_g_state[base + "mlp.c_proj.weight"]
            if base + "mlp.c_proj.bias" in clip_g_state:
                mapped[hf_base + "mlp.fc2.bias"] = clip_g_state[base + "mlp.c_proj.bias"]

            # layer norms
            if base + "ln_1.weight" in clip_g_state:
                mapped[hf_base + "layer_norm1.weight"] = clip_g_state[base + "ln_1.weight"]
            if base + "ln_1.bias" in clip_g_state:
                mapped[hf_base + "layer_norm1.bias"] = clip_g_state[base + "ln_1.bias"]
            if base + "ln_2.weight" in clip_g_state:
                mapped[hf_base + "layer_norm2.weight"] = clip_g_state[base + "ln_2.weight"]
            if base + "ln_2.bias" in clip_g_state:
                mapped[hf_base + "layer_norm2.bias"] = clip_g_state[base + "ln_2.bias"]

        # embeddings
        if "token_embedding.weight" in clip_g_state:
            mapped["text_model.embeddings.token_embedding.weight"] = clip_g_state["token_embedding.weight"]
        if "positional_embedding" in clip_g_state:
            pe = clip_g_state["positional_embedding"]
            if isinstance(pe, torch.Tensor):
                mapped["text_model.embeddings.position_embedding.weight"] = pe.detach().clone()
            else:
                mapped["text_model.embeddings.position_embedding.weight"] = torch.tensor(pe)
        if "ln_final.weight" in clip_g_state:
            mapped["text_model.final_layer_norm.weight"] = clip_g_state["ln_final.weight"]
        if "ln_final.bias" in clip_g_state:
            mapped["text_model.final_layer_norm.bias"] = clip_g_state["ln_final.bias"]

        # 检测 intermediate_size
        inter_size = None
        for idx in sorted(layer_indices):
            k = f"text_model.encoder.layers.{idx}.mlp.fc1.weight"
            if k in mapped:
                inter_size = mapped[k].shape[0]
                break

        overrides = {"num_hidden_layers": n_layers}
        if hidden_size:
            overrides["hidden_size"] = hidden_size
        if inter_size:
            overrides["intermediate_size"] = inter_size

        return mapped, overrides

    def load(self) -> bool:
        if self._loaded:
            return True
        try:
            import safetensors.torch
        except ImportError:
            return False

        state = self._load_or_extract()

        # 分离 CLIP-L 和 CLIP-G
        clip_l = {}
        clip_g_raw = {}
        for k, v in state.items():
            if k.startswith("clip_l."):
                clip_l[k[len("clip_l."):]] = v
            elif k.startswith("clip_g."):
                clip_g_raw[k[len("clip_g."):]] = v

        if not clip_l and not clip_g_raw:
            raise RuntimeError("未找到 CLIP 权重 (预期 clip_l.* 和 clip_g.* 键)")

        # 转换 CLIP-G
        clip_g_mapped, g_overrides = self._convert_clip_g(clip_g_raw)

        # CLIP-L: 自动检测架构
        l_layers = set()
        l_hidden = None
        l_inter = None
        for k in clip_l:
            if "encoder.layers." in k:
                idx_str = k.split("encoder.layers.")[1].split(".")[0]
                if idx_str.isdigit():
                    l_layers.add(int(idx_str))
        n_layers_l = max(l_layers) + 1 if l_layers else 12
        for k, v in clip_l.items():
            if k.endswith(".self_attn.q_proj.weight"):
                l_hidden = v.shape[0]
            if k.endswith(".mlp.fc1.weight"):
                l_inter = v.shape[0]
            if l_hidden and l_inter:
                break

        # 构建 config overrides（与默认 JSON 合并）
        l_overrides = {"num_hidden_layers": n_layers_l}
        if l_hidden:
            l_overrides["hidden_size"] = l_hidden
        if l_inter:
            l_overrides["intermediate_size"] = l_inter

        dtype = torch.float32
        device = self._device

        from engines.comfy_clip.sdxl_clip import SDXLClipModel, SDXLTokenizer

        model_options = {
            "clip_l_model_config": l_overrides,
            "clip_g_model_config": g_overrides,
        }

        self._model = SDXLClipModel(
            device=device, dtype=dtype,
            model_options=model_options
        )
        self._model.to(device)

        # CLIP-L: 直接加载 (已是 HF/ComfyUI text_model.* 格式)
        missing_l, unexpected_l = self._model.clip_l.load_sd(clip_l)
        if missing_l:
            print(f"[CLIP-L] 缺失键 (前5): {list(missing_l)[:5]}")

        # CLIP-G: 加载转换后的权重
        missing_g, unexpected_g = self._model.clip_g.load_sd(clip_g_mapped)
        if missing_g:
            print(f"[CLIP-G] 缺失键 (前5): {list(missing_g)[:5]}")

        self._model.eval()
        self._tokenizer = SDXLTokenizer()
        self._loaded = True
        return True

    def encode(self, text: str) -> np.ndarray:
        """编码文本为 2048 维向量（与 ComfyUI 向量库 100% 一致）。"""
        if not self._loaded:
            raise RuntimeError("CLIP 模型未加载")

        device = self._device

        # 1. Tokenize
        tw = self._tokenizer.tokenize_with_weights(text)

        # 2. Encode (move token tensors to device happens inside process_tokens)
        with torch.no_grad():
            out, pooled = self._model.encode_token_weights(tw)

        # 3. Mean pool across sequence → 2048-dim
        vec = out[0].mean(dim=0).cpu().numpy().astype(np.float32)

        # 4. L2 normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec


_LOCAL_CLIP = None


def get_encoder(ckpt_path: str = None) -> LocalSDXLEncoder:
    global _LOCAL_CLIP
    if ckpt_path is None:
        return None
    ckpt_path = str(ckpt_path)
    if _LOCAL_CLIP is not None and str(_LOCAL_CLIP.ckpt_path) == ckpt_path:
        return _LOCAL_CLIP
    _LOCAL_CLIP = LocalSDXLEncoder(ckpt_path)
    return _LOCAL_CLIP


def encode_text_local(text: str, ckpt_path: str) -> np.ndarray:
    encoder = get_encoder(ckpt_path)
    if encoder is None:
        raise RuntimeError("未指定 CLIP 模型路径")
    if not encoder._loaded:
        encoder.load()
    return encoder.encode(text)