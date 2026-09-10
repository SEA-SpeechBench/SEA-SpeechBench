from src.dataset_src.speechbench.base_dataset import BaseDataset
from src.dataset_src.speechbench.registry import register_dataset

class BaseGRDataset(BaseDataset):
    task_category = "PQA"
    task_type = "GR"
    metric_name = "Gemma3-27B-Instruct_Binary"

@register_dataset("gr_ytb_eval_human_30")
class gr_ytb_eval_human_30(BaseGRDataset):
    dataset_name = "gr_ytb_eval_human_30"

    def __init__(self, split="test_subset1000", num_sample=-1, num_worker=4, use_diverse_prompts=False, use_sea_prompts=False):
        super().__init__(split, num_sample, num_worker, use_diverse_prompts, use_sea_prompts)
        # dataset has its own prompts
        self.use_diverse_prompts = False

def register_gr_dataset():
    """
    Registers dataset classes with the global DATASET_REGISTRY.

    The registered dataset class will be used to create a dataset instance
    when `get_dataset` is called with the registered name.
    """
    def create_dataset_class(name):
        @register_dataset(name)
        class _Dataset(BaseGRDataset):
            dataset_name = name
        # clean up class name
        _Dataset.__name__ = name
        return _Dataset

    for dataset_name in [
        "gr_cv21_id_30",
        "gr_cv21_ta_30",
        "gr_cv21_th_30",
        "gr_cv21_vi_30",
        "gr_cv21_zh_30",
        "gr_emota_ta_30",
        "gr_fleurs_en_30",
        "gr_fleurs_km_30",
        "gr_indowave_id_30",
        "gr_m3ed_30",
        "gr_openslr_ta_30",
        "gr_sg_streets_utterance_30",
        "gr_smaldusc_30",
        "gr_thai_elderly_th_30",
        "gr_thai_ser_th_30",
        "gr_vietnam_celeb_30",
    ]:
        create_dataset_class(dataset_name)

    return


