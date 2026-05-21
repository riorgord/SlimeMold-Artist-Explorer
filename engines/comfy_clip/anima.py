# Adapted from ComfyUI (GPL-3.0) — comfy/text_encoders/anima.py. See NOTICE in this directory.
from pathlib import Path
from transformers import Qwen2Tokenizer
from . import sd1_clip
from . import llama
from . import ops


class Qwen3Tokenizer(sd1_clip.SDTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        tokenizer_path = str(Path(__file__).resolve().parent.parent / "tokenizers" / "qwen3")
        super().__init__(
            tokenizer_path,
            pad_with_end=False,
            embedding_directory=embedding_directory,
            embedding_size=1024, embedding_key='qwen3_06b',
            tokenizer_class=Qwen2Tokenizer,
            has_start_token=False, has_end_token=False,
            pad_to_max_length=False, max_length=99999999,
            min_length=1, pad_token=151643,
            tokenizer_data=tokenizer_data
        )


class AnimaTokenizer:
    """Anima tokenizer — Qwen3 only (no T5XXL needed for vector encoding)."""
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        self.qwen3_06b = Qwen3Tokenizer(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data)

    def tokenize_with_weights(self, text: str, return_word_ids=False, **kwargs):
        out = {}
        qwen_ids = self.qwen3_06b.tokenize_with_weights(text, return_word_ids, **kwargs)
        out["qwen3_06b"] = [
            [(k[0], 1.0) for k in inner_list]
            for inner_list in qwen_ids
        ]
        return out

    def untokenize(self, token_weight_pair):
        return self.qwen3_06b.untokenize(token_weight_pair)

    def state_dict(self):
        return {}


class Qwen3_06BModel(sd1_clip.SDClipModel):
    def __init__(self, device="cpu", layer="last", layer_idx=None, dtype=None, attention_mask=True, model_options={}):
        super().__init__(
            device=device, layer=layer, layer_idx=layer_idx,
            textmodel_json_config={}, dtype=dtype,
            special_tokens={"pad": 151643},
            layer_norm_hidden_state=False,
            model_class=llama.Qwen3_06B,
            enable_attention_masks=attention_mask,
            return_attention_masks=attention_mask,
            model_options=model_options
        )