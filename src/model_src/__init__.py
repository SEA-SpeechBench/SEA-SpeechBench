import torch
import importlib
from abc import ABC, abstractmethod
from src.model_src import registry as model_registry
from src.model_src.registry import register_model, MODEL_REGISTRY


class BaseModel(ABC):
    MAX_TIME_LIMIT = None  # in seconds

    def __init__(self, model_name_or_path):
        self.model_name = model_name_or_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_model()

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def generate(self, input, generation_config=None):
        pass


def create_model_class(registry_model_name, load_model_func, generate_func, max_time_limit):
    @register_model(registry_model_name)
    class _Model(BaseModel):
        MAX_TIME_LIMIT = max_time_limit

        @torch.no_grad()
        def load_model(self):
            load_model_func(self)

        @torch.no_grad()
        def generate(self, input):
            return generate_func(self, input)

    _Model.__name__ = registry_model_name
    return _Model


MODEL_SRC = "src.model_src"
MODEL_MAP = {
    "Qwen2.5-Omni-3B": {
        "model_file": "qwen2_5_omni_3B",
        "max_time_limit": 60
    },
    "Qwen2.5-Omni-7B": {
        "model_file": "qwen2_5_omni_7B",
        "max_time_limit": 120
    },
    "MERaLiON-2-10B": {
        "model_file": "meralion2_10b",
        "max_time_limit": 300
    },
    "MERaLiON-2-3B": {
        "model_file": "meralion2_3b",
        "max_time_limit": 120
    },
    "Phi-4-multimodal-instruct": {
        "model_file": "phi_4_multimodal_instruct",
        "max_time_limit": 120
    },
    "Qwen2-Audio-7B-Instruct": {
        "model_file": "qwen2_audio_7b_instruct",
        "max_time_limit": 30
    },
    "SeaLLMs-Audio-7B": {
        "model_file": "seallms_audio_7b",
        "max_time_limit": 30
    },
    "Kimi-Audio-7B-Instruct": {
        "model_file": "kimi_audio_7b_instruct",
        "max_time_limit": 60
    },
    "gpt-4o-audio-preview": {
        "model_file": "gpt4o_audio_preview",
        "max_time_limit": 1500
    },
    "gpt-4o-transcribe": {
        "model_file": "gpt4o_transcribe",
        "max_time_limit": 1500
    },
    "gemini-2.5-pro": {
        "model_file": "gemini_2_5_pro",
        "max_time_limit": 1500
    },
    "gemini-2.5-flash": {
        "model_file": "gemini_2_5_flash",
        "max_time_limit": 1500
    },
    "Qwen3-Omni-30B-A3B-Instruct": {
        "model_file": "qwen3_omni_30b_a3b_instruct",
        "max_time_limit": 600
    },
    "Qwen3-Omni-30B-A3B-Thinking": {
        "model_file": "qwen3_omni_30b_a3b_thinking",
        "max_time_limit": 900
    },
    "cascade-whisper-large-v3-turbo-Qwen3.6-35B-A3B": {
        "model_file": "cascade_whisper_large_v3_turbo_qwen3_6_35b_a3b",
        "max_time_limit": 300
    },
}


def _registry_name(model_name: str) -> str:
    return model_name.replace("-", "_").lower()


def ensure_model_registered(model_name: str) -> str:
    """Import and register only the requested model (avoids loading all backends)."""
    registry_model_name = _registry_name(model_name)
    if registry_model_name in MODEL_REGISTRY:
        return registry_model_name

    matched = None
    for map_name, model_item in MODEL_MAP.items():
        if _registry_name(map_name) == registry_model_name:
            matched = (map_name, model_item)
            break
    if matched is None:
        raise ValueError(f"Model {model_name} not found in MODEL_MAP")

    _, model_item = matched
    model_file = model_item["model_file"]
    max_time_limit = model_item["max_time_limit"]
    module = importlib.import_module(f"{MODEL_SRC}.{model_file}")
    load_model_func = getattr(module, f"{model_file}_model_loader")
    generate_func = getattr(module, f"{model_file}_model_generation")
    create_model_class(registry_model_name, load_model_func, generate_func, max_time_limit)
    return registry_model_name


def get_model(model_name):
    ensure_model_registered(model_name)
    return MODEL_REGISTRY[_registry_name(model_name)](model_name)


# main_evaluate imports get_model from registry; keep that entrypoint working.
model_registry.get_model = get_model
