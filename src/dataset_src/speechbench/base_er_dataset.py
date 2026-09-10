from src.dataset_src.speechbench.base_dataset import BaseDataset
from src.dataset_src.speechbench.registry import register_dataset

class BaseERDataset(BaseDataset):
    task_category = "PQA"
    task_type = "ER"
    # metric_name = "Gemma3-27B-Instruct_Binary"
    metric_name = "Qwen3-Omni-30B-A3B-Instruct_Binary"

@register_dataset("er_ytb_eval_human_30")
class er_ytb_eval_human_30(BaseERDataset):
    dataset_name = "er_ytb_eval_human_30"

    def __init__(self, split="test_subset1000", num_sample=-1, num_worker=4, use_diverse_prompts=False, use_sea_prompts=False):
        super().__init__(split, num_sample, num_worker, use_diverse_prompts, use_sea_prompts)
        # dataset has its own prompts
        self.use_diverse_prompts = False

def register_er_dataset():
    """
    Registers dataset classes with the global DATASET_REGISTRY.

    The registered dataset class will be used to create a dataset instance
    when `get_dataset` is called with the registered name.
    """
    def create_dataset_class(name):
        @register_dataset(name)
        class _Dataset(BaseERDataset):
            dataset_name = name
        # clean up class name
        _Dataset.__name__ = name
        return _Dataset

    for dataset_name in [
        "er_emota_ta_30",
        "er_esd_en_30",
        "er_esd_zh_30",
        "er_indowave_id_30",
        "er_m3ed_30",
        "er_tec_ta_30",
        "er_thai_ser_th_30"
    ]:
        create_dataset_class(dataset_name)

    return