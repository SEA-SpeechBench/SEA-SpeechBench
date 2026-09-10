import random
import numpy as np

from src.dataset_src.speechbench.base_dataset import BaseDataset
from src.dataset_src.speechbench.registry import register_dataset

def generate_beep(frequency=440, duration=0.5, sampling_rate=16000, volume=0.5):
    """Generates a sine wave beep."""
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    beep = volume * np.sin(2 * np.pi * frequency * t)
    return beep.astype(np.float64)

class BaseSpRDataset(BaseDataset):
    task_category = "SpR"
    task_type = "SpR"

    # metric_name = "Gemma3-27B-Instruct_Binary"
    metric_name = "Qwen3-Omni-30B-A3B-Instruct_Binary"
    
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
        beep = generate_beep()
        audio_length_beep = len(beep)/16000
        audio = [
            {
                "array": np.concatenate([
                    x["audio_1"]["array"],
                    beep,
                    x["audio_2"]["array"]
                ]),
                "sampling_rate": 16000
            }
            for x in batch["context"]
        ]
        if self.use_diverse_prompts:
            instruction = random.choices(self.prompts, k=len(audio))
        else:
            assert self.use_sea_prompts == False, "SEA prompts are only available with diverse prompts"
            instruction = [x["text"] for x in batch["instruction"]]

        reference = [x["text"] for x in batch["answer"]]
        language = [x for x in batch["language_1"]]
        audio_duration = [x + batch["audio_length_2"][i] + audio_length_beep for i, x in enumerate(batch["audio_length_1"])]

        return {
            "audio": audio,
            "text": instruction,
            "answer": reference,
            "task_type": [self.task_type] * len(audio),
            "language": language,
            "audio_duration": audio_duration,
        }

def register_spr_dataset():
    """
    Registers dataset classes with the global DATASET_REGISTRY.

    The registered dataset class will be used to create a dataset instance
    when `get_dataset` is called with the registered name.
    """
    def create_dataset_class(name):
        @register_dataset(name)
        class _Dataset(BaseSpRDataset):
            dataset_name = name
        # clean up class name
        _Dataset.__name__ = name
        return _Dataset

    for dataset_name in [
        "spr_emota_ta_30",
        "spr_esd_en_30",
        "spr_esd_zh_30",
        "spr_mig_my_30",
        "spr_smaldusc_30",
        "spr_thai_elderly_th_30",
        "spr_thai_ser_30",
        "spr_voxvietnam_vi_30",
    ]:
        create_dataset_class(dataset_name)

    return