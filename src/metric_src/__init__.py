import torch
import importlib
from abc import ABC, abstractmethod
from .registry import register_metric

class BaseMetric(ABC):

    def __init__(self, metric_name, vllm_port):
        """
        Initialize the metric.

        Args:
            metric_name (str): Name of the metric.
            vllm_port (int): Port number of the VLLM server.
        """
        self.metric_name = metric_name
        self.vllm_port = vllm_port

    @abstractmethod
    def evaluate(self, input_data):
        pass

def create_metric_class(registry_metric_name, evaluate_func):
    @register_metric(registry_metric_name)
    class _Metric(BaseMetric):
        @torch.no_grad()
        def evaluate(self, input):
            return evaluate_func(self, input)

    # clean up class name
    _Metric.__name__ = registry_metric_name

    return _Metric


# Metrics

METRIC_SRC = "src.metric_src"
METRIC_MAP = {
    "Gemma3-27B-Instruct_General": {
        "metric_file": "gemma3_27b_instruct",
        "evaluation_func": "gemma3_27b_instruct_general_judge"
    },
    "Gemma3-27B-Instruct_Binary": {
        "metric_file": "gemma3_27b_instruct",
        "evaluation_func": "gemma3_27b_instruct_binary_judge"
    },
    "Qwen3-Omni-30B-A3B-Instruct_General": {
        "metric_file": "qwen3_omni_30b_a3b_instruct",
        "evaluation_func": "qwen3_omni_30b_a3b_instruct_general_judge"
    },
    "Qwen3-Omni-30B-A3B-Instruct_Binary": {
        "metric_file": "qwen3_omni_30b_a3b_instruct",
        "evaluation_func": "qwen3_omni_30b_a3b_instruct_binary_judge"
    },
    "BLEU": {
        "metric_file": "bleu",
        "evaluation_func": "bleu"
    },
    "F1_Coverage_Purity": {
        "metric_file": "f1_coverage_purity",
        "evaluation_func": "f1_coverage_purity_evaluate"
    },
    # Commented out ASR metrics since asr_metrics module has import issues
    # "WER": {
    #     "metric_file": "asr_metrics",
    #     "evaluation_func": "adaptive_evaluate"
    # },
    # "CER": {
    #     "metric_file": "asr_metrics", 
    #     "evaluation_func": "adaptive_evaluate"
    # },
    # "ASR_TCQ": {
    #     "metric_file": "asr_metrics",
    #     "evaluation_func": "adaptive_evaluate"
    # }
}

for metric_name, metric in METRIC_MAP.items():
    metric_file_path = f"{METRIC_SRC}.{metric['metric_file']}"
    module = importlib.import_module(metric_file_path)
    evaluate_func = getattr(module, metric["evaluation_func"])

    # format metric_name
    registry_metric_name = metric_name.replace("-", "_").lower()

    # register metric
    create_metric_class(registry_metric_name, evaluate_func)

