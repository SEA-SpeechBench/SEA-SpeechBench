import random
import re

from src.dataset_src.speechbench.base_dataset import BaseDataset
from src.dataset_src.speechbench.registry import register_dataset

class BaseTLocDataset(BaseDataset):
    task_category = "TLoc"
    task_type = "TLoc"
    metric_name = "F1_Coverage_Purity"

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
        audio = [x["audio"] for x in batch["context"]]
        if self.use_diverse_prompts:
            instruction = random.choices(self.prompts, k=len(audio))
            # get query transcript
            query_transcript = [x["text"].split(":")[-1].strip() for x in batch["instruction"]]
            # replace query transcription placeholder
            instruction = [
                instruction_i.replace("QUERY_TRANSCRIPTION", query_transcript[i])
                for i, instruction_i in enumerate(instruction)
            ]
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

def register_tloc_dataset():
    """
    Registers dataset classes with the global DATASET_REGISTRY.

    The registered dataset class will be used to create a dataset instance
    when `get_dataset` is called with the registered name.
    """
    def create_dataset_class(name):
        @register_dataset(name)
        class _Dataset(BaseTLocDataset):
            dataset_name = name
        # clean up class name
        _Dataset.__name__ = name
        return _Dataset

    dataset_list = [
        "tloc_sgpccsc_long_30",
        "tloc_sgpccsc_long_60",
        "tloc_sgpccsc_long_120",
        "tloc_sgpccsc_long_180",
        "tloc_sg_streets_30",
        "tloc_sg_streets_60",
    ]
    for name in ["en", "id", "th", "vi", "zh"]:
        for duration in [30, 60, 120, 180]:
            dataset_list.append(f"tloc_yodas2_{name}_{duration}")

    for dataset_name in dataset_list:
        create_dataset_class(dataset_name)

    return