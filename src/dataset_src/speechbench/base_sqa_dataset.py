from src.dataset_src.speechbench.base_dataset import BaseDataset
from src.dataset_src.speechbench.registry import register_dataset

class BaseSQADataset(BaseDataset):
    task_category = "SQA"
    task_type = "SQA"
    # metric_name = "Gemma3-27B-Instruct_General"
    metric_name = "Qwen3-Omni-30B-A3B-Instruct_General"

    def __init__(self, split="test_subset1000", num_sample=-1, num_worker=4, use_diverse_prompts=False, use_sea_prompts=False):
        super().__init__(split, num_sample, num_worker, use_diverse_prompts, use_sea_prompts)
        # dataset has its own prompts
        self.use_diverse_prompts = False

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
        assert self.use_diverse_prompts == False

        if self.use_sea_prompts:
            instruction = [x["text"] for x in batch["instruction"]]
            reference = [x["text"] for x in batch["answer"]]
        else:
            instruction = [x["english_question"] for x in batch["other_attributes"]]
            reference = [x["english_answer"] for x in batch["other_attributes"]]

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

def register_sqa_dataset():
    """
    Registers dataset classes with the global DATASET_REGISTRY.

    The registered dataset class will be used to create a dataset instance
    when `get_dataset` is called with the registered name.
    """
    def create_dataset_class(name):
        @register_dataset(name)
        class _Dataset(BaseSQADataset):
            dataset_name = name
        # clean up class name
        _Dataset.__name__ = name
        return _Dataset

    for dataset_name in [
        "sqa_ytb_eval_human_30",
        "sqa_sgpccsc_30",
        "sqa_sg_streets_30",
        "sqa_yodas2_en_30",
        "sqa_yodas2_id_30",
        "sqa_yodas2_th_30",
        "sqa_yodas2_vi_30",
        "sqa_yodas2_zh_30"
    ]:
        create_dataset_class(dataset_name)

    return