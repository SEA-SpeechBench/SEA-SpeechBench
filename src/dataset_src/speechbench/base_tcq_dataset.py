import random
import re

from src.dataset_src.speechbench.base_dataset import BaseDataset
from src.dataset_src.speechbench.registry import register_dataset

class BaseTCQDataset(BaseDataset):
    task_category = "TCQ"
    task_type = "TCQ"
    metric_name = "ASR_TCQ"

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
            language_placeholder = self.code2lang[self.language][self.language]
        else:
            "Please help me transcribe the speech into text in Chinese."
            language_placeholder = self.code2lang[self.language]["en"]

        audio = [x["audio"] for x in batch["context"]]
        if self.use_diverse_prompts:
            instruction = random.choices(self.prompts, k=len(audio))
            # replace language placeholder
            instruction = [instruction_i.replace("LANGUAGE", language_placeholder) for instruction_i in instruction]
            # get query start time and end time
            query_start_time, query_end_time = zip(*[
                tuple(map(float, re.search(r'\[(\d+\.?\d*),\s*(\d+\.?\d*)\]', x['text']).groups()))
                for x in batch["instruction"]
            ])
            # replace query start time and end time placeholder
            instruction = [
                instruction_i.replace("QUERY_START_TIME", str(query_start_time[i])).replace("QUERY_END_TIME", str(query_end_time[i]))
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

def register_tcq_dataset():
    """
    Registers dataset classes with the global DATASET_REGISTRY.

    The registered dataset class will be used to create a dataset instance
    when `get_dataset` is called with the registered name.
    """
    def create_dataset_class(name):
        @register_dataset(name)
        class _Dataset(BaseTCQDataset):
            dataset_name = name
        # clean up class name
        _Dataset.__name__ = name
        return _Dataset

    dataset_list = [
        "tcq_sgpccsc_30",
        "tcq_sgpccsc_60",
        "tcq_sgpccsc_120",
        "tcq_sgpccsc_180",
        "tcq_sg_streets_30",
        "tcq_sg_streets_60",
    ]
    for name in ["en", "id", "th", "vi", "zh"]:
        for duration in [30, 60, 120, 180]:
            dataset_list.append(f"tcq_yodas2_{name}_{duration}")

    for dataset_name in dataset_list:
        create_dataset_class(dataset_name)

    return