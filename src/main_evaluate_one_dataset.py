
import os

# Force set cache directories BEFORE importing any other modules
os.environ['HF_HOME'] = _expand('${HF_HOME}')
os.environ['TRANSFORMERS_CACHE'] = _expand('${HF_HOME}')
os.environ['HF_DATASETS_CACHE'] = _expand('${HF_DATASETS_CACHE}')

import fire
import json
import logging
import torch
from tqdm import tqdm
from typing import Optional

from src.dataset_src.speechbench.registry import get_dataset
from src.model_src.registry import get_model
from os.path import expandvars as _expand

# =  =  =  =  =  =  =  =  =  =  =  Logging Setup  =  =  =  =  =  =  =  =  =  =  =  =  = 
logger = logging.getLogger(__name__)
logging.basicConfig(
    format  = "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt = "%m/%d/%Y %H:%M:%S",
    level   = logging.INFO,
)
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  = 

def do_model_prediction(input_data, model, batch_size, generation_config=None):
    """
    Do model prediction on a list of input data.

    Args:
        input_data (list): List of input data to be predicted.
        model (Model): The model to be used for prediction.
        batch_size (int): The batch size for prediction. Note that this is not used currently.
        generation_config (dict, optional): Configuration for generation. Defaults to None.

    Returns:
        list: The list of predictions.
    """
    assert batch_size == 1, "Batch size {} not supported yet".format(batch_size)

    model_predictions = []
    for inputs in tqdm(input_data, leave=False):
        # TODO model.generate does not support generation_config yet
        # outputs = model.generate(inputs, generation_config=generation_config)
        outputs = model.generate(inputs)
        if isinstance(outputs, list):
            model_predictions.extend(outputs)
        else:
            model_predictions.append(outputs)

    return model_predictions


def main(
        dataset_name        : str  = None,
        model_name          : str  = None,
        batch_size          : int  = 1,     # it is now a dummy parameter
        split               : str  = "test_subset1000-short",
        use_diverse_prompts : bool = True,
        use_sea_prompts     : bool = False,
        generation_config   : Optional[dict] = None,
        overwrite           : bool = False,
        number_of_samples   : int  = -1,
        out_dir             : str  = "log",
        vllm_port           : Optional[int] = None,
        ):

    """
    Evaluate a model on a dataset.

    Args:
        dataset_name (str): The name of the dataset to evaluate.
        model_name (str): The name of the model to evaluate.
        batch_size (int): The batch size to use. Currently, a dummy parameter as it is not used.
        split (str): The split of the dataset to evaluate.
        use_diverse_prompts (bool): Whether to use diverse prompts or not.
        use_sea_prompts (bool): Whether to use SEA prompts or not.
        generation_config (dict): The configuration for generation.
        overwrite (bool): Whether to overwrite the results or not.
        number_of_samples (int): The number of samples to evaluate.
        out_dir (str): The directory to save the results.

    Returns:
        None
    """

    logger.info("= = "*20)
    logger.info("Dataset name: {}".format(dataset_name))
    logger.info("Model name: {}".format(model_name))
    logger.info("Batch size: {}".format(batch_size))
    logger.info("Split: {}".format(split))
    logger.info("Use diverse prompts: {}".format(use_diverse_prompts))
    logger.info("Use sea prompts: {}".format(use_sea_prompts))
    logger.info("Generation_config: {}".format(generation_config))
    logger.info("Overwrite: {}".format(overwrite))
    logger.info("Number of samples: {}".format(number_of_samples))
    logger.info("Out dir: {}".format(out_dir))
    logger.info("= = "*20)

    # check that proprietary models are running on smaller split
    if "gpt" in model_name.lower() or "gemini" in model_name.lower():
        assert split in ["test_subset100-short", "test_subset100-long"], "Proprietary models should be evaluated on the smaller test_subset100 split."
    # check that ASR models are running on ASR tasks
    if model_name.lower() in ["gpt-4o-transcribe"]:
        assert "asr_" in dataset_name.lower(), "ASR models should be evaluated on ASR datasets."
        use_diverse_prompts = False
        use_sea_prompts = False
    elif model_name.lower() in ["meralion-2-10b-asr"]:
        assert "asr_" in dataset_name.lower(), "ASR models should be evaluated on ASR datasets."
    elif model_name.lower() in ["gpt-4o-audio-preview"]:
        assert "asr_" not in dataset_name.lower(), "ASR datasets should be evaluated with ASR versions of models."

    generation_config = None if not isinstance(generation_config, dict) else generation_config
    model_savename = "{}-{}".format(model_name, "-".join(f"{k}{v}" for k, v in generation_config.items())) \
        if generation_config is not None else model_name

    # Load dataset and get updated use_diverse_prompts and use_sea_prompts
    logger.info("Preparing dataset: {}".format(dataset_name))
    dataset = get_dataset(
        split = split,
        name = dataset_name,
        num_sample = number_of_samples,
        use_diverse_prompts = use_diverse_prompts,
        use_sea_prompts = use_sea_prompts
    )
    task_category = dataset.task_category
    logger.info("Prepared {} samples for evaluation".format(dataset.num_sample))
    logger.info("= = "*20)

    use_diverse_prompts = dataset.use_diverse_prompts
    use_sea_prompts = dataset.use_sea_prompts
    model_savename = "{}_prompts-{}-diverse{}".format(model_savename, dataset.language if use_sea_prompts else "en", use_diverse_prompts)
    os.makedirs(os.path.join(out_dir, split, task_category, model_savename), exist_ok=True)

    # If the final score log exists, skip the evaluation
    if not overwrite and os.path.exists('{}/{}/{}/{}/{}_score.json'.format(
        out_dir, split, task_category, model_savename, dataset_name)):
        logger.info("Evaluation has been done before. Skip the evaluation.")
        logger.info("\n\n\n\n\n")
        return

    # Infer with model
    if overwrite or not os.path.exists('{}/{}/{}/{}/{}.json'.format(
        out_dir, split, task_category, model_savename, dataset_name)):
        logger.info("Overwrite is enabled or the results are not found. Try to infer with the model: {}.".format(model_savename))

        # Prepare model input
        dataset.input_data = dataset.prepare_model_input()

        # Load model
        logger.info("Preparing model: {}".format(model_name))
        model = get_model(model_name)
        logger.info("Loaded model: {}".format(model_name))
        logger.info("= = "*20)

        # Check if the model can handle the audio length
        max_audio_length = int(dataset.dataset_name.split("_")[-1])
        if model.MAX_TIME_LIMIT < max_audio_length:
            logger.info(f"The model cannot handle dataset of duration {max_audio_length}s. Skip the evaluation.")
            logger.info("\n\n\n\n\n")
            return

        # Infer with model
        model_predictions           = do_model_prediction(dataset.input_data, model, batch_size=batch_size, generation_config=generation_config)
        data_with_model_predictions = dataset.format_model_predictions(dataset.input_data, model_predictions)

        # Save the result with predictions
        with open('{}/{}/{}/{}/{}.json'.format(out_dir, split, task_category, model_savename, dataset_name), 'w') as f:
            json.dump(data_with_model_predictions, f, indent=4, ensure_ascii=False)

    data_with_model_predictions = json.load(open('{}/{}/{}/{}/{}.json'.format(out_dir, split, task_category, model_savename, dataset_name)))

    # Metric evaluation
    try:
        # Clear the cache to avoid memory leak
        logger.info("Clear the cache to avoid memory leak")
        del model
        torch.cuda.empty_cache()
    except: 
        pass

    # Check if the metric requires VLLM
    # BLEU and other objective metrics don't need VLLM
    metric_requires_vllm = dataset.metric_name in ["Gemma3-27B-Instruct_General", "Gemma3-27B-Instruct_Binary", "Qwen3-Omni-30B-A3B-Instruct_General", "Qwen3-Omni-30B-A3B-Instruct_Binary"]
    
    if vllm_port is None and metric_requires_vllm:
        # Try to get vllm_port from environment variable
        vllm_port = os.environ.get('MY_VLLM_PORT_JUDGE')
    
        if vllm_port:
            try:
                vllm_port = int(vllm_port)
                logger.info(f"Using VLLM port from environment variable: {vllm_port}")
            except (ValueError, TypeError):
                logger.warning(f"Invalid VLLM port in environment variable: {vllm_port}")
                vllm_port = None
        
        # if vllm_port is None:
        #     logger.info("VLLM port is required for this metric but not provided. Skip the evaluation.")
        #     return
    logger.info("VLLM port checked: {}".format(vllm_port))

    # try:
    results = dataset.compute_score(data_with_model_predictions, vllm_port)

    # Print the result
    logger.info('=  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =')
    logger.info('Dataset name: {}'.format(dataset_name.upper()))
    logger.info('Model name: {}'.format(model_savename.upper()))
    logger.info(json.dumps({dataset.metric_name: results["metric_scores"]}, indent=4, ensure_ascii=False))
    logger.info('=  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =')

    # Save the scores
    with open('{}/{}/{}/{}/{}_score.json'.format(out_dir, split, task_category, model_savename, dataset_name, dataset.metric_name), 'w') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    fire.Fire(main)
