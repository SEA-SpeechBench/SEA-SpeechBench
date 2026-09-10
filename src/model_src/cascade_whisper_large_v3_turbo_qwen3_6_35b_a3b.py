#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""Cascade: Whisper-large-v3-turbo ASR -> Qwen3.6-35B-A3B MT for SEA ST.

Whisper uses the container transformers (compatible with system torch/torchvision).
Qwen3.6 temporarily uses ./qwen36_packages transformers@main only while loading/running MT.
"""

import gc
import logging
import os
import re
import sys

sys.path.append(".")
sys.path.append("../")

os.environ["HF_ENDPOINT"] = "https://huggingface.co"

import numpy as np
import torch
from huggingface_hub import snapshot_download
from os.path import expandvars as _expand

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

WHISPER_REPO = os.environ.get("WHISPER_MODEL_PATH", "openai/whisper-large-v3-turbo")
QWEN_REPO = os.environ.get("QWEN_MODEL_PATH", "Qwen/Qwen3.6-35B-A3B")
QWEN36_PACKAGES = os.path.abspath(
    os.environ.get(
        "QWEN36_PACKAGES",
        _expand("${DATA_ROOT}/qwen36_packages"),
    )
)

MT_PROMPT_TEMPLATE = """\
[Audio Transcription]
{transcript}

[Instruction]
{instruction}

Translate the transcription according to the instruction above.
Output only the translation text, with no explanation or extra formatting.
"""

# Modules that must be reloaded from qwen36_packages when switching to Qwen.
_QWEN36_MODULE_PREFIXES = (
    "transformers",
    "accelerate",
    "huggingface_hub",
    "safetensors",
    "tokenizers",
)


def _strip_thinking(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<\|thinking\|>.*?<\|/thinking\|>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def _local_snapshot(repo_id: str) -> str:
    try:
        return snapshot_download(repo_id=repo_id, local_files_only=True)
    except Exception:
        logger.warning("Local cache miss for %s; downloading...", repo_id)
        return snapshot_download(repo_id=repo_id)


def _audio_array(input_data):
    audio = input_data["audio"]
    if isinstance(audio, dict):
        return np.asarray(audio["array"], dtype=np.float32), int(audio.get("sampling_rate", 16000))
    return np.asarray(audio, dtype=np.float32), 16000


def _model_device(model):
    return next(model.parameters()).device


def _purge_qwen36_modules():
    for key in list(sys.modules):
        if any(key == p or key.startswith(p + ".") for p in _QWEN36_MODULE_PREFIXES):
            del sys.modules[key]


def _enter_qwen36_transformers():
    """Prefer qwen36_packages transformers stack; keep system torch."""
    pkg = os.path.abspath(QWEN36_PACKAGES)
    if not os.path.isdir(pkg):
        raise FileNotFoundError(f"QWEN36_PACKAGES not found: {pkg}")
    while pkg in sys.path:
        sys.path.remove(pkg)
    sys.path.insert(0, pkg)
    _purge_qwen36_modules()

    import huggingface_hub
    import safetensors
    from safetensors import safe_open

    hub_file = os.path.abspath(huggingface_hub.__file__)
    st_file = os.path.abspath(safetensors.__file__)
    if pkg not in hub_file:
        raise RuntimeError(f"Expected huggingface_hub from {pkg}, got {hub_file}")
    if pkg not in st_file:
        raise RuntimeError(f"Expected safetensors from {pkg}, got {st_file}")
    # New transformers passes backend= to safe_open; old system safetensors lacks it.
    if "backend" not in getattr(safe_open, "__text_signature__", "(filename, framework, device, *, backend)"):
        # __text_signature__ may be absent on some builds; probe via TypeError path instead.
        pass
    logger.info("Using huggingface_hub from %s", hub_file)
    logger.info("Using safetensors from %s (%s)", st_file, getattr(safetensors, "__version__", "?"))


def _leave_qwen36_transformers():
    pkg = os.path.abspath(QWEN36_PACKAGES)
    while pkg in sys.path:
        sys.path.remove(pkg)
    _purge_qwen36_modules()


def _load_whisper(self):
    # Container transformers + system torch (do NOT use qwen36_packages here).
    pkg = os.path.abspath(QWEN36_PACKAGES)
    while pkg in sys.path:
        sys.path.remove(pkg)
    _purge_qwen36_modules()
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    whisper_path = getattr(self, "whisper_local_path", None) or _local_snapshot(WHISPER_REPO)
    self.whisper_local_path = whisper_path
    logger.info("Loading Whisper from %s", whisper_path)

    self.whisper_processor = WhisperProcessor.from_pretrained(
        whisper_path, local_files_only=True
    )
    self.whisper_model = WhisperForConditionalGeneration.from_pretrained(
        whisper_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        local_files_only=True,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    self.whisper_model.to(device)
    self.whisper_model.eval()


def _load_qwen(self):
    _enter_qwen36_transformers()
    # Qwen3.6-35B-A3B is qwen3_5_moe multimodal MoE; checkpoint architecture is
    # Qwen3_5MoeForConditionalGeneration (not a plain CausalLM).
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoTokenizer

    qwen_path = getattr(self, "qwen_local_path", None) or _local_snapshot(QWEN_REPO)
    self.qwen_local_path = qwen_path
    logger.info("Loading Qwen from %s (transformers from %s)", qwen_path, QWEN36_PACKAGES)

    # transformers@main bug with composite Qwen3.5/3.6 MoE configs: missing-key
    # re-init reads config.initializer_range, but it only lives under text_config.
    config = AutoConfig.from_pretrained(
        qwen_path, trust_remote_code=True, local_files_only=True
    )
    ir = 0.02
    text_cfg = getattr(config, "text_config", None)
    if text_cfg is not None and getattr(text_cfg, "initializer_range", None) is not None:
        ir = text_cfg.initializer_range
    try:
        object.__setattr__(config, "initializer_range", ir)
    except Exception:
        config.initializer_range = ir
    logger.info("Patched config.initializer_range=%s", ir)

    # Also patch model class _init_weights path in case config object is replaced.
    import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as qwen_moe_mod

    _orig_init_weights = qwen_moe_mod.Qwen3_5MoePreTrainedModel._init_weights

    def _init_weights_with_ir(self, module):
        cfg = self.config
        if not hasattr(cfg, "initializer_range"):
            _ir = ir
            _tc = getattr(cfg, "text_config", None)
            if _tc is not None and getattr(_tc, "initializer_range", None) is not None:
                _ir = _tc.initializer_range
            try:
                object.__setattr__(cfg, "initializer_range", _ir)
            except Exception:
                cfg.initializer_range = _ir
        return _orig_init_weights(self, module)

    qwen_moe_mod.Qwen3_5MoePreTrainedModel._init_weights = _init_weights_with_ir

    self.llm_tokenizer = AutoTokenizer.from_pretrained(
        qwen_path,
        trust_remote_code=True,
        padding_side="left",
        local_files_only=True,
    )
    if self.llm_tokenizer.pad_token is None:
        self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token

    load_kwargs = dict(
        config=config,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    try:
        self.llm_model = AutoModelForImageTextToText.from_pretrained(
            qwen_path, dtype=torch.bfloat16, **load_kwargs
        )
    except TypeError as e:
        # Only fall back for dtype kwarg mismatches, not safetensors/backend errors.
        if "dtype" not in str(e) and "torch_dtype" not in str(e):
            raise
        self.llm_model = AutoModelForImageTextToText.from_pretrained(
            qwen_path, torch_dtype=torch.bfloat16, **load_kwargs
        )
    except ValueError as e:
        if "accelerate" in str(e).lower():
            raise RuntimeError(
                "Qwen3.6 needs accelerate in qwen36_packages (without bundling a second torch)."
            ) from e
        raise

    # Belt-and-suspenders: model.config may be a fresh object without the patch.
    if not hasattr(self.llm_model.config, "initializer_range"):
        try:
            object.__setattr__(self.llm_model.config, "initializer_range", ir)
        except Exception:
            self.llm_model.config.initializer_range = ir

    self.llm_model.eval()


def _unload_whisper(self):
    for attr in ("whisper_model", "whisper_processor"):
        if hasattr(self, attr):
            delattr(self, attr)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _unload_qwen(self):
    for attr in ("llm_model", "llm_tokenizer"):
        if hasattr(self, attr):
            delattr(self, attr)
    _leave_qwen36_transformers()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _whisper_asr(self, input_data) -> str:
    audio_array, sampling_rate = _audio_array(input_data)
    language = input_data.get("language")

    inputs = self.whisper_processor(
        audio_array,
        sampling_rate=sampling_rate,
        return_tensors="pt",
    )
    input_features = inputs.input_features.to(
        device=_model_device(self.whisper_model),
        dtype=self.whisper_model.dtype,
    )

    gen_kwargs = {}
    if language:
        try:
            gen_kwargs["forced_decoder_ids"] = self.whisper_processor.get_decoder_prompt_ids(
                language=language, task="transcribe"
            )
        except Exception:
            pass

    predicted_ids = self.whisper_model.generate(input_features, **gen_kwargs)
    text = self.whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return text.strip()


def _qwen_mt(self, transcript: str, instruction: str) -> str:
    user_content = MT_PROMPT_TEMPLATE.format(
        transcript=transcript,
        instruction=instruction,
    )
    messages = [{"role": "user", "content": user_content}]

    try:
        prompt = self.llm_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        prompt = self.llm_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            chat_template_kwargs={"enable_thinking": False},
        )

    device = _model_device(self.llm_model)
    encoded = self.llm_tokenizer(prompt, return_tensors="pt", padding=True)
    encoded = {k: v.to(device) for k, v in encoded.items()}

    generated_ids = self.llm_model.generate(
        **encoded,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        do_sample=True,
        pad_token_id=self.llm_tokenizer.eos_token_id,
    )
    generated_ids = generated_ids[:, encoded["input_ids"].shape[-1] :]
    text = self.llm_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return _strip_thinking(text)


def cascade_whisper_large_v3_turbo_qwen3_6_35b_a3b_model_loader(self):
    self.whisper_local_path = _local_snapshot(WHISPER_REPO)
    self.qwen_local_path = _local_snapshot(QWEN_REPO)

    # keep_qwen: load Qwen once after first ASR; only swap Whisper (small) per sample.
    # Legacy "sequential" reloads Qwen every sample (too slow for full runs).
    self.cascade_mode = os.environ.get("CASCADE_MODE", "keep_qwen")
    self._qwen_ready = False
    _load_whisper(self)
    logger.info(
        "Cascade mode=%s ready. Whisper=%s Qwen=%s",
        self.cascade_mode,
        self.whisper_local_path,
        self.qwen_local_path,
    )


def cascade_whisper_large_v3_turbo_qwen3_6_35b_a3b_model_generation(self, input_data):
    if input_data.get("task_type") != "ST":
        raise NotImplementedError(
            f"Cascade Whisper+Qwen3.6 currently supports task_type=ST only, "
            f"got {input_data.get('task_type')!r}."
        )

    instruction = input_data["text"]

    if self.cascade_mode == "joint":
        if not hasattr(self, "whisper_model"):
            _load_whisper(self)
        if not getattr(self, "_qwen_ready", False):
            _load_qwen(self)
            self._qwen_ready = True
        transcript = _whisper_asr(self, input_data)
        logger.info("ASR transcript: %s", transcript[:200])
        return _qwen_mt(self, transcript, instruction)

    # keep_qwen (default) / sequential
    if not hasattr(self, "whisper_model"):
        _load_whisper(self)
    transcript = _whisper_asr(self, input_data)
    logger.info("ASR transcript: %s", transcript[:200])
    _unload_whisper(self)

    if not getattr(self, "_qwen_ready", False):
        _load_qwen(self)
        self._qwen_ready = True
    try:
        translation = _qwen_mt(self, transcript, instruction)
    finally:
        if self.cascade_mode == "sequential":
            _unload_qwen(self)
            self._qwen_ready = False
            _load_whisper(self)
        else:
            # keep_qwen: reload Whisper only for the next sample's ASR
            _load_whisper(self)
    return translation
