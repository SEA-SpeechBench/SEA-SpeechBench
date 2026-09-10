from src.dataset_src.speechbench.base_age_dataset import register_age_dataset
from src.dataset_src.speechbench.base_asr_dataset import register_asr_dataset
from src.dataset_src.speechbench.base_er_dataset import register_er_dataset
from src.dataset_src.speechbench.base_gr_dataset import register_gr_dataset
from src.dataset_src.speechbench.base_sqa_dataset import register_sqa_dataset
from src.dataset_src.speechbench.base_spr_dataset import register_spr_dataset
from src.dataset_src.speechbench.base_st_dataset import register_st_dataset
from src.dataset_src.speechbench.base_tcq_dataset import register_tcq_dataset
from src.dataset_src.speechbench.base_tloc_dataset import register_tloc_dataset

__all__ = [
    "register_age_dataset",
    "register_asr_dataset",
    "register_er_dataset",
    "register_gr_dataset",
    "register_sqa_dataset",
    "register_spr_dataset",
    "register_st_dataset",
    "register_tcq_dataset",
    "register_tloc_dataset",
]

register_age_dataset()
register_asr_dataset()
register_er_dataset()
register_gr_dataset()
register_sqa_dataset()
register_spr_dataset()
register_st_dataset()
register_tcq_dataset()
register_tloc_dataset()

