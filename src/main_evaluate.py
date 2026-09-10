import os

# Force set cache directories BEFORE importing any other modules
os.environ['HF_HOME'] = _expand('${HF_HOME}')
os.environ['TRANSFORMERS_CACHE'] = _expand('${HF_HOME}')
os.environ['HF_DATASETS_CACHE'] = _expand('${HF_DATASETS_CACHE}')

import fire
import torch
import subprocess
from itertools import islice, cycle
from queue import Queue
from threading import Thread
from src.dataset_src.speechbench.registry import DATASET_REGISTRY
from os.path import expandvars as _expand

# CONFIG
SCRIPT = "src.main_evaluate_one_dataset"
DATASETS = list(DATASET_REGISTRY.keys())

NUM_GPUS = min(4, torch.cuda.device_count())

def worker(gpu_id, queue, model_name, use_diverse_prompts, use_sea_prompts, out_dir, vllm_port):

    split_prefix = "test_subset1000"
    if "gpt" in model_name.lower() or "gemini" in model_name.lower():
        split_prefix = "test_subset100"

    while not queue.empty():
        dataset = queue.get()
        max_audio_length = int(dataset.split("_")[-1])
        split_suffix = "short" if max_audio_length <= 30 else "long"
        split = f"{split_prefix}-{split_suffix}"

        print(f"[GPU {gpu_id}] Starting {dataset}")
        cmd = (
            f"CUDA_VISIBLE_DEVICES={gpu_id} "
            f"python -m {SCRIPT} "
            f"--vllm_port {vllm_port} "
            f"--dataset_name {dataset} "
            f"--use_diverse_prompts {use_diverse_prompts} "
            f"--use_sea_prompts {use_sea_prompts} "
            f"--model_name {model_name} "
            f"--split {split} "
            f"--batch_size 1 "
            f"--overwrite False "
            f"--number_of_samples -1 "
            f"--out_dir {out_dir} "
            f"2> {out_dir}/{dataset}.stderr.txt 1> {out_dir}/{dataset}.stdout.txt"
        )
        # Pass environment variables to subprocess
        env = os.environ.copy()
        # Ensure MY_VLLM_PORT_JUDGE is set for the subprocess
        env['MY_VLLM_PORT_JUDGE'] = str(vllm_port)
        subprocess.run(cmd, shell=True, env=env)
        print(f"[GPU {gpu_id}] Finished {dataset}")
        queue.task_done()

def repeat_to_length(vllm_port_list_str, NUM_GPUS):
    vllm_port_list_expanded = list(islice(cycle(vllm_port_list_str.split("n")), NUM_GPUS))
    vllm_port_list_int = map(int, vllm_port_list_expanded)
    vllm_port_list = sorted(vllm_port_list_int)
    return vllm_port_list

def main(
    model_name          : str  = None,
    use_diverse_prompts : bool = True,
    use_sea_prompts     : bool = False,
    out_dir             : str  = "log",
    vllm_port_list_str  : str = None,
):
    queue = Queue()

    vllm_port_list = repeat_to_length(vllm_port_list_str, NUM_GPUS) if type(vllm_port_list_str) == str else [vllm_port_list_str]*NUM_GPUS
    assert len(vllm_port_list) == NUM_GPUS

    # check that ASR models are running on ASR tasks
    if model_name.lower() in ["gpt-4o-transcribe", "meralion-2-10b-asr"]:
        datasets_filtered = [d for d in DATASETS if "asr_" in d.lower()]
    elif model_name.lower() in ["gpt-4o-audio-preview"]:
        datasets_filtered = [d for d in DATASETS if "asr_" not in d.lower()]
    else:
        datasets_filtered = DATASETS

    # Custom filtering for testing specific tasks
    # Uncomment and modify the following lines to test only specific tasks:
    
    # Test only TLoc tasks:
    # datasets_filtered = [d for d in datasets_filtered if "tloc_" not in d.lower()]
    
    # Test only specific datasets:
    # datasets_filtered = [d for d in datasets_filtered if d in ["tloc_sg_streets_30", "tloc_sgpccsc_30"]]
    
    # Test only short duration tasks (30 seconds):
    # datasets_filtered = [d for d in datasets_filtered if d.endswith("_30")]

    for ds in datasets_filtered:
        queue.put(ds)

    threads = []
    for gpu in range(NUM_GPUS):
        t = Thread(
            target=worker,
            args=(gpu, queue, model_name, use_diverse_prompts, use_sea_prompts, out_dir, vllm_port_list[gpu]),
        )
        t.start()
        threads.append(t)

    queue.join()
    for t in threads:
        t.join()

if __name__ == "__main__":
    fire.Fire(main)
