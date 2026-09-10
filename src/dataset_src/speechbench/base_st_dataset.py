import random
import re

from src.dataset_src.speechbench.base_dataset import BaseDataset
from src.dataset_src.speechbench.registry import register_dataset

class BaseSTDataset(BaseDataset):
    task_category = "ST"
    task_type = "ST"
    metric_name = "BLEU"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_language = self.raw_data[0]["language_target"]

    def _reformat(self, batch):
        """
        Reformat the batch of dataset to match the required format.

        The required format is:
        {
            "audio": [...],
            "text": [...],
            "answer": [...],
            "task_type": [...],
            "language": [...],
            "audio_duration": [...]
        }

        Args:
            batch: a batch of dataset

        Returns:
            a batch of dataset in the required format
        """
        if self.use_sea_prompts:
            "请帮我把这段语音转写成中文文本。"
            target_language_placeholder = self.code2lang[self.target_language][self.language]
        else:
            "Please help me transcribe the speech into text in Chinese."
            target_language_placeholder = self.code2lang[self.target_language]["en"]

        audio = [x["audio"] for x in batch["context"]]
        if self.use_diverse_prompts:
            instruction = random.choices(self.prompts, k=len(audio))
            # replace target language placeholder
            instruction = [instruction_i.replace("TARGET_LANGUAGE", target_language_placeholder) for instruction_i in instruction]
        else:
            assert self.use_sea_prompts == False, "SEA prompts are only available with diverse prompts"
            instruction = [x["text"] for x in batch["instruction"]]

        reference = [x["text"] for x in batch["answer"]]
        language = [x for x in batch["language"]]
        audio_duration = [x for x in batch["audio_length"]]

        return {
            "audio": audio,
            "text": instruction,
            "answer": reference,
            "task_type": [self.task_type] * len(audio),
            "language": language,
            "audio_duration": audio_duration,
        }

def register_st_dataset():
    """
    Registers dataset classes with the global DATASET_REGISTRY.

    The registered dataset class will be used to create a dataset instance
    when `get_dataset` is called with the registered name.
    """
    def create_dataset_class(name):
        @register_dataset(name)
        class _Dataset(BaseSTDataset):
            dataset_name = name
        # clean up class name
        _Dataset.__name__ = name
        return _Dataset

    dataset_list = [
    ]
    for name in ["id", "km", "lo", "ms", "my", "th", "tl", "vi", "zh"]:
        dataset_list.append(f"translation_fleurs_{name}_30")

    for dataset_name in dataset_list:
        create_dataset_class(dataset_name)

    return