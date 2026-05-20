"""
本地 CLIP 编码器 —— 从 SDXL checkpoint 提取 CLIP 权重，本地编码文本。
不依赖 ComfyUI。需要: pip install torch safetensors transformers

SDXL checkpoint key 结构:
  CLIP-L: conditioner.embedders.0.transformer.text_model.*  (HuggingFace格式)
  CLIP-G: conditioner.embedders.1.model.*                    (OpenCLIP格式)
  UNet:   model.diffusion_model.*
  VAE:    first_stage_model.*
"""

import numpy as np
from pathlib import Path


def _short_hash(path: Path) -> str:
    size = path.stat().st_size
    return f"{path.stem}_{size}"


class LocalCLIPEncoder:
    def __init__(self, ckpt_path: str, cache_dir: str = "./data/cached_clip"):
        self.ckpt_path = Path(ckpt_path)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._clip_l = None
        self._clip_g = None
        self._tokenizer_l = None
        self._tokenizer_g = None
        self._loaded = False

    def _is_full_checkpoint(self, state: dict) -> bool:
        for k in state:
            if k.startswith("model.diffusion_model."):
                return True
        return False

    def _has_conditioner(self, state: dict) -> bool:
        """检测是否使用 conditioner.embedders 格式（SDXL）"""
        for k in state:
            if k.startswith("conditioner.embedders."):
                return True
        return False

    def _load_or_extract(self) -> dict:
        """加载 CLIP 权重。首次从完整 checkpoint 全量加载提取并缓存，之后走缓存。"""
        import safetensors.torch
        cache_name = f"{_short_hash(self.ckpt_path)}_clip.safetensors"
        cache_path = self.cache_dir / cache_name

        # 有缓存直接用
        if cache_path.exists():
            return safetensors.torch.load_file(str(cache_path))

        # 首次：全量加载（safe_open 的 get_tensor 在大文件上偶发 NaN）
        print("🔧 首次加载: 从 checkpoint 提取 CLIP 权重...")
        full_state = safetensors.torch.load_file(str(self.ckpt_path))

        # 提取 CLIP-L 和 CLIP-G
        clip_state = {}
        for k, v in full_state.items():
            if k.startswith("conditioner.embedders.0.transformer."):
                clip_state[f"clip_l.{k[len('conditioner.embedders.0.transformer.'):]}"] = v
            elif k.startswith("conditioner.embedders.1.model."):
                clip_state[f"clip_g.{k[len('conditioner.embedders.1.model.'):]}"] = v
        del full_state  # 立即释放 6GB

        if not clip_state:
            raise ValueError("未找到 conditioner.embedders 格式的 CLIP 权重")

        safetensors.torch.save_file(clip_state, str(cache_path))
        print(f"💾 CLIP 权重已缓存: {cache_path}  ({cache_path.stat().st_size // 1024 // 1024}MB)")
        return safetensors.torch.load_file(str(cache_path))

    @staticmethod
    def _split_in_proj(in_proj_weight, in_proj_bias=None):
        """将 OpenCLIP 的 in_proj [3*hidden, hidden] 拆为 q/k/v_proj [hidden, hidden]"""
        import torch
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
    def _convert_clip_g_to_hf(clip_g_state: dict):
        """将 OpenCLIP 格式的 CLIP-G 权重转为 HuggingFace CLIPTextModel 格式。
        返回 (mapped_dict, num_layers)。"""
        import torch
        mapped = {}
        # 文本投影和温度
        if "text_projection" in clip_g_state:
            mapped["text_projection"] = clip_g_state["text_projection"]
        if "logit_scale" in clip_g_state:
            mapped["logit_scale"] = clip_g_state["logit_scale"]

        # 遍历 resblocks → encoder.layers，检测层数
        prefix = "transformer.resblocks."
        layer_indices = set()
        for k in clip_g_state:
            if k.startswith(prefix):
                idx_str = k[len(prefix):].split(".")[0]
                if idx_str.isdigit():
                    layer_indices.add(int(idx_str))
        n_layers = max(layer_indices) + 1 if layer_indices else 24

        for layer_idx in sorted(layer_indices):
            base = f"transformer.resblocks.{layer_idx}."
            hf_base = f"text_model.encoder.layers.{layer_idx}."

            # in_proj → split into q,k,v (key 格式: in_proj_weight / in_proj_bias)
            w_key = base + "attn.in_proj_weight"
            b_key = base + "attn.in_proj_bias"
            if w_key in clip_g_state:
                w = clip_g_state[w_key]
                b = clip_g_state.get(b_key)
                parts = LocalCLIPEncoder._split_in_proj(w, b)
                for sub_k, sub_v in parts.items():
                    mapped[hf_base + "self_attn." + sub_k] = sub_v
                # out_proj
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
        # final layer norm
        if "ln_final.weight" in clip_g_state:
            mapped["text_model.final_layer_norm.weight"] = clip_g_state["ln_final.weight"]
        if "ln_final.bias" in clip_g_state:
            mapped["text_model.final_layer_norm.bias"] = clip_g_state["ln_final.bias"]

        return mapped, n_layers

    def load(self) -> bool:
        if self._loaded:
            return True
        try:
            import torch
            import safetensors.torch
        except ImportError:
            return False

        # 用 safe_open 增量加载（只读 CLIP 部分，不载入完整 checkpoint）
        try:
            state = self._load_or_extract()
        except Exception as e:
            raise RuntimeError(f"无法读取模型文件: {e}")

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

        # CLIP-L: 直接加载 (已是 HuggingFace text_model.* 格式)
        from transformers import CLIPTextModel, CLIPTextConfig, CLIPTokenizer
        # 从 checkpoint 自动检测层数
        l_layers = set()
        for k in clip_l:
            if "encoder.layers." in k:
                idx_str = k.split("encoder.layers.")[1].split(".")[0]
                if idx_str.isdigit():
                    l_layers.add(int(idx_str))
        n_layers_l = max(l_layers) + 1 if l_layers else 12
        cfg_l = CLIPTextConfig(
            vocab_size=49408, hidden_size=768, intermediate_size=3072,
            num_hidden_layers=n_layers_l, num_attention_heads=12,
            max_position_embeddings=77,
        )
        self._clip_l = CLIPTextModel(cfg_l)
        mapped_l = {}
        for k, v in clip_l.items():
            # text_model.embeddings.xxx → embeddings.xxx (CLIPTextModel 内部不带前缀)
            if k.startswith("text_model."):
                mapped_l[k[len("text_model."):]] = v
            else:
                mapped_l[k] = v
        if mapped_l:
            missing, unexpected = self._clip_l.load_state_dict(mapped_l, strict=False)
            if missing:
                print(f"[CLIP-L] 缺失键 (前5): {list(missing)[:5]}")
        else:
            raise RuntimeError("CLIP-L 未找到有效权重")

        # CLIP-G: 转换 OpenCLIP → HuggingFace 格式
        mapped_g, n_layers_g = self._convert_clip_g_to_hf(clip_g_raw)
        cfg_g = CLIPTextConfig(
            vocab_size=49408, hidden_size=1280, intermediate_size=5120,
            num_hidden_layers=n_layers_g, num_attention_heads=20,
            max_position_embeddings=77,
        )
        self._clip_g = CLIPTextModel(cfg_g)
        # CLIPTextModel 内部参数名不带 text_model. 前缀，统一去掉
        mapped_g = {k[len("text_model."):] if k.startswith("text_model.") else k: v
                    for k, v in mapped_g.items()}
        if mapped_g:
            missing_g, unexpected_g = self._clip_g.load_state_dict(mapped_g, strict=False)
            if missing_g:
                print(f"[CLIP-G] 缺失键 (前5): {list(missing_g)[:5]}")
        else:
            raise RuntimeError("CLIP-G 未找到有效权重")

        # tokenizers
        _tk_dir = Path(__file__).resolve().parent / "tokenizers"
        self._tokenizer_l = CLIPTokenizer.from_pretrained(str(_tk_dir / "tokenizer_clip-vit-large-patch14"))
        self._tokenizer_g = CLIPTokenizer.from_pretrained(str(_tk_dir / "CLIP-ViT-bigG-14-laion2B-39B-b160k"))

        self._loaded = True
        return True

    def encode(self, text: str) -> np.ndarray:
        import torch
        if not self._loaded:
            raise RuntimeError("CLIP 模型未加载")

        self._clip_l.eval()
        self._clip_g.eval()

        tok_l = self._tokenizer_l(text, return_tensors="pt", padding="max_length",
                                  max_length=77, truncation=True)
        with torch.no_grad():
            out_l = self._clip_l(input_ids=tok_l["input_ids"],
                                 attention_mask=tok_l["attention_mask"],
                                 output_hidden_states=True)
        # 用倒数第二层 (layer_idx=-2)，和 ComfyUI 一致
        pooled_l = out_l.hidden_states[-2].mean(dim=1)[0].numpy().astype(np.float32)

        tok_g = self._tokenizer_g(text, return_tensors="pt", padding="max_length",
                                  max_length=77, truncation=True)
        with torch.no_grad():
            out_g = self._clip_g(input_ids=tok_g["input_ids"],
                                 attention_mask=tok_g["attention_mask"],
                                 output_hidden_states=True)
        pooled_g = out_g.hidden_states[-2].mean(dim=1)[0].numpy().astype(np.float32)

        combined = np.concatenate([pooled_l, pooled_g]).astype(np.float32)
        norm = np.linalg.norm(combined)
        if norm > 0:
            combined /= norm
        return combined


_LOCAL_CLIP = None


def get_encoder(ckpt_path: str = None) -> LocalCLIPEncoder:
    global _LOCAL_CLIP
    if ckpt_path is None:
        return None
    ckpt_path = str(ckpt_path)
    if _LOCAL_CLIP is not None and str(_LOCAL_CLIP.ckpt_path) == ckpt_path:
        return _LOCAL_CLIP
    _LOCAL_CLIP = LocalCLIPEncoder(ckpt_path)
    return _LOCAL_CLIP


def encode_text_local(text: str, ckpt_path: str) -> np.ndarray:
    encoder = get_encoder(ckpt_path)
    if encoder is None:
        raise RuntimeError("未指定 CLIP 模型路径")
    if not encoder._loaded:
        encoder.load()
    return encoder.encode(text)
