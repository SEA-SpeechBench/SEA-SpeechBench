import os

# Force set cache directories BEFORE importing transformers
os.environ['HF_HOME'] = _expand('${HF_HOME}')
os.environ['TRANSFORMERS_CACHE'] = _expand('${HF_HOME}')
os.environ['HF_DATASETS_CACHE'] = _expand('${HF_DATASETS_CACHE}')

import logging
import torch
import numpy as np
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from os.path import expandvars as _expand

# =  =  =  =  =  =  =  =  =  =  =  Logging Setup  =  =  =  =  =  =  =  =  =  =  =  =  =
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

model_path = "MERaLiON/MERaLiON-2-10B"

def meralion2_10b_model_loader(self):
    
    # # Ensure cache directories exist
    # cache_dir = _expand('${HF_HOME}')
    # os.makedirs(cache_dir, exist_ok=True)
    # os.makedirs(f'{cache_dir}/datasets', exist_ok=True)
    # os.makedirs(f'{cache_dir}/modules', exist_ok=True)
    # os.makedirs(f'{cache_dir}/modules/transformers_modules', exist_ok=True)

    self.processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_path,
        use_safetensors=True,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16
    ).to(self.device)
    logger.info("Model loaded: {}".format(model_path))


def meralion2_10b_model_generation(self, input_data):

    audio_array     = input_data["audio"]["array"]
    sampling_rate   = input_data["audio"]["sampling_rate"]
    task_type       = input_data["task_type"]
    instruction     = input_data["text"]

    prompt_template = "Instruction: {query} \nFollow the text instruction based on the following audio: <SpeechHere>"

    # Create messages with system prompt and audio input
    conversation = [
        [
            {
                "role": "user",
                "content": prompt_template.format(query=instruction)
            }
        ],
    ]

    chat_prompt = self.processor.tokenizer.apply_chat_template(
        conversation=conversation,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = self.processor(text=chat_prompt, audios=[np.asarray(audio_array)])

    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            inputs[key] = inputs[key].to(self.device)

            if value.dtype == torch.float32:
                inputs[key] = inputs[key].to(torch.bfloat16)

    outputs = self.model.generate(**inputs, max_new_tokens=256)
    generated_ids = outputs[:, inputs['input_ids'].size(1):]
    response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    return response