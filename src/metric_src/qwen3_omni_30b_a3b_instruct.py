
import os
import socket
import time
import logging
import transformers
from tqdm import tqdm
from openai import OpenAI
from multiprocessing import Pool
from src.metric_src.prompts.model_judge_prompts import GENERAL_TEMPLATE, BINARY_TEMPLATE

logger = logging.getLogger(__name__)

# Try to import Qwen3 Omni Processor if available
try:
    from transformers import Qwen3OmniProcessor
    QWEN3_OMNI_AVAILABLE = True
except ImportError:
    try:
        from transformers import Qwen3OmniForConditionalGeneration, Qwen3OmniProcessor
        QWEN3_OMNI_AVAILABLE = True
    except ImportError:
        QWEN3_OMNI_AVAILABLE = False
        Qwen3OmniProcessor = None


def qwen3_omni_30b_a3b_instruct_one_sample(args):
    """
    Evaluate one sample using model judge.

    Args:
    - processor: The processor to use for formatting the sample.
    - prompt_template: The template for the evaluation prompt.
    - question: The question to evaluate.
    - reference: The reference answer.
    - prediction: The model's prediction.

    Returns:
    - A dictionary containing the sample's rating details.
    """

    processor, prompt_template, vllm_port, question, reference, prediction = args
    
    # Truncate inputs more aggressively to avoid context length issues
    max_length = 500  # Reduced to 500 characters
    if len(question) > max_length:
        question = question[:max_length] + "..."
    if len(reference) > max_length:
        reference = reference[:max_length] + "..."
    if len(prediction) > max_length:
        prediction = prediction[:max_length] + "..."

    evaluation_prompt = prompt_template.format(question=question, prediction=prediction, reference=reference)

    messages = [
        {"role": "user", "content": evaluation_prompt},
    ]

    # Check if processor has chat_template, if not use a simple format
    try:
        templated_sample = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            tokenize=False,
        )
    except (ValueError, AttributeError):
        # Fallback to simple template if chat_template is not available
        templated_sample = f"<|im_start|>user\n{evaluation_prompt}<|im_end|>\n<|im_start|>assistant\n"

    # Model
    openai_api_key = "EMPTY"
    openai_api_base = f"http://localhost:{vllm_port}/v1"
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
        timeout=30.0,  # 30 second timeout
        max_retries=3,  # Retry up to 3 times
    )

    try:
        models = client.models.list()
        model = models.data[0].id

        completion = client.completions.create(
            model      = model,
            prompt     = templated_sample,
            max_tokens = 128,
            n          = 1,
        )
        
        output = completion.choices[0].text.strip()
    except Exception as e:
        logger.error(f"API call failed in qwen3_omni_30b_a3b_instruct: {e}")
        output = "Error: API call failed"

    try:
        # Extract the score, handling potential special tokens
        # Try to find "Rating: X" pattern first
        import re
        rating_match = re.search(r'[Rr]ating:\s*(\d+(?:\.\d+)?)', output)
        if rating_match:
            rate_score = float(rating_match.group(1))
        else:
            # Fallback: get last token and clean special tokens
            last_token = output.split()[-1]
            # Remove common special tokens
            last_token = last_token.replace('<|im_end|>', '').replace('<|endoftext|>', '').replace('</s>', '').strip()
            rate_score = float(last_token)
        success = 1
    except:
        rate_score = 0.0
        success = 0

    sample_rating_detail = {
        'question'        : question,
        'reference'       : reference,
        'model_prediction': prediction,
        'judge_response'  : output,
        'rate_score'      : rate_score,
        'success'         : success,
    }

    return sample_rating_detail


def qwen3_omni_30b_a3b_instruct(input_data, prompt_template, vllm_port):
    """
    Evaluate using model judge.

    This function evaluates the model predictions against reference answers.
    Each sample is processed to compute a score based on accuracy and relevance,
    using the provided prompt template.

    Args:
        input_data (tuple): A tuple consisting of questions, references, and predictions.
        prompt_template (str): A template for generating evaluation prompts.

    Returns:
        tuple: A dictionary with the average judge score and success rate, and
               a list containing detailed evaluation results for each sample.

    """
    judge_path = "Qwen/Qwen3-Omni-30B-A3B-Instruct"

    # Load processor - try Qwen3OmniProcessor first, fallback to AutoProcessor
    if QWEN3_OMNI_AVAILABLE and Qwen3OmniProcessor is not None:
        processor = Qwen3OmniProcessor.from_pretrained(judge_path)
    else:
        processor = transformers.AutoProcessor.from_pretrained(judge_path, use_fast=True)

    # Generation
    questions, references, predictions = input_data

    num_processes = min(24, len(questions))

    with Pool(processes=num_processes) as pool:
        all_details = list(
            tqdm(
                pool.imap(
                    qwen3_omni_30b_a3b_instruct_one_sample,
                    zip([processor]*len(questions), [prompt_template]*len(questions), [vllm_port]*len(questions), \
                        questions, references, predictions)
                ),
                total=len(questions),
                desc="Processing samples with model judge"
            )
        )

    return all_details

def wait_for_vllm(port, host="localhost", timeout=600):
    """Wait for vLLM server to be ready.
    
    Increased timeout to 600 seconds (10 minutes) for large models like Qwen3-Omni-30B
    which may take longer to start.

    Returns:
        True if ready within timeout, False otherwise.
    """
    start_time = time.time()
    logger.info(f"Waiting for vLLM server at {host}:{port} (timeout: {timeout}s)...")
    while time.time() - start_time < timeout:
        try:
            with socket.create_connection((host, port), timeout=2):
                logger.info(f"vLLM server at {host}:{port} is ready!")
                return True  # Ready
        except (ConnectionRefusedError, socket.timeout):
            elapsed = int(time.time() - start_time)
            if elapsed % 30 == 0:  # Log every 30 seconds
                logger.info(f"Still waiting for vLLM server... ({elapsed}s elapsed)")
            time.sleep(1)
    logger.error(f"vLLM server at {host}:{port} not ready after {timeout}s")
    return False  # Not ready in time

def qwen3_omni_30b_a3b_instruct_general_judge(self, input_data):

    if not wait_for_vllm(self.vllm_port):
        print(f"[ERROR] vLLM at port {self.vllm_port} is not ready.")
        raise Exception("vLLM server not ready")
    all_details = qwen3_omni_30b_a3b_instruct(input_data, GENERAL_TEMPLATE, self.vllm_port)

    all_scores   = [detail['rate_score'] * 20 for detail in all_details]
    avg_score    = sum(all_scores) / len(all_scores)
    success_rate = sum([detail['success'] for detail in all_details]) / len(all_details)

    judge_results = {'judge_score': avg_score, 'success_rate': success_rate}

    return judge_results, all_details

def qwen3_omni_30b_a3b_instruct_binary_judge(self, input_data):

    if not wait_for_vllm(self.vllm_port):
        print(f"[ERROR] vLLM at port {self.vllm_port} is not ready.")
        raise Exception("vLLM server not ready")
    all_details = qwen3_omni_30b_a3b_instruct(input_data, BINARY_TEMPLATE, self.vllm_port)

    all_scores   = [detail['rate_score'] * 100 for detail in all_details]
    avg_score    = sum(all_scores) / len(all_scores)
    success_rate = sum([detail['success'] for detail in all_details]) / len(all_details)

    judge_results = {'judge_score': avg_score, 'success_rate': success_rate}

    return judge_results, all_details