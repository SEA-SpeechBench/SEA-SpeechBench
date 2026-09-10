from src.dataset_src.speechbench.base_dataset import BaseDataset
from src.dataset_src.speechbench.registry import register_dataset

class BaseAgeDataset(BaseDataset):
    task_category = "PQA"
    task_type = "AgeP"
    metric_name = "Gemma3-27B-Instruct_Binary"

def register_age_dataset():
    """
    Registers dataset classes with the global DATASET_REGISTRY.

    The registered dataset class will be used to create a dataset instance
    when `get_dataset` is called with the registered name.
    """
    def create_dataset_class(name):
        @register_dataset(name)
        class _Dataset(BaseAgeDataset):
            dataset_name = name
        # clean up class name
        _Dataset.__name__ = name
        return _Dataset

    for dataset_name in [
        "age_cv21_en_30",
        "age_cv21_ta_30",
        "age_cv21_th_30",
        "age_cv21_vi_30",
        "age_cv21_zh_30"
    ]:
        create_dataset_class(dataset_name)

    return