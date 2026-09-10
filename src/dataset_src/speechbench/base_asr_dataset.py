import random

from src.dataset_src.speechbench.base_dataset import BaseDataset
from src.dataset_src.speechbench.registry import register_dataset

class BaseASRDataset(BaseDataset):
    task_category = "ASR"
    task_type = "ASR"
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
            language_placeholder = self.code2lang[self.language][self.language]
        else:
            language_placeholder = self.code2lang[self.language]["en"]

        audio = [x["audio"] for x in batch["context"]]
        if self.use_diverse_prompts:
            instruction = random.choices(self.prompts, k=len(audio))
            # replace language placeholder
            instruction = [instruction_i.replace("LANGUAGE", language_placeholder) for instruction_i in instruction]
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

def register_asr_dataset():
    """
    Registers dataset classes with the global DATASET_REGISTRY.

    The registered dataset class will be used to create a dataset instance
    when `get_dataset` is called with the registered name.
    """
    def create_dataset_class(name):
        @register_dataset(name)
        class _Dataset(BaseASRDataset):
            dataset_name = name
        # clean up class name
        _Dataset.__name__ = name
        return _Dataset

    dataset_list = [
        "asr_bloomspeech_en_30",
        "asr_bloomspeech_my_30",
        "asr_bud500_30",
        "asr_cv21_en_30",
        "asr_cv21_id_30",
        "asr_cv21_ta_30",
        "asr_cv21_th_30",
        "asr_cv21_vi_30",
        "asr_cv21_zh_30",
        "asr_esd_en_30",
        "asr_esd_zh_30",
        "asr_fleurs_fil_30",
        "asr_fleurs_id_30",
        "asr_fleurs_km_30",
        "asr_fleurs_lo_30",
        "asr_fleurs_ms_30",
        "asr_fleurs_my_30",
        "asr_fleurs_th_30",
        "asr_fleurs_vi_30",
        "asr_fleurs_zh_30",
        "asr_malcsc_30",
        "asr_mig_my_30",
        "asr_openslr_km_30",
        "asr_openslr_my_30",
        "asr_openslr_ta_30",
        "asr_sfdusc_30",
        "asr_sgpccsc_utterance_30",
        "asr_sg_streets_utterance_30",
        "asr_smaldusc_30",
        "asr_thai_elderly_th_30",
        "asr_thai_lotus_30",
        "asr_vietmed_vi_30",
    ]

    for dataset_name in dataset_list:
        create_dataset_class(dataset_name)

    return