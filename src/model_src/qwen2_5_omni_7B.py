
# add parent directory to sys.path
import sys
sys.path.append('.')
sys.path.append('../')
import logging
import numpy as np
import torch
from tqdm import tqdm
try:
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
    QWEN_AVAILABLE = True
except ImportError:
    QWEN_AVAILABLE = False
    Qwen2_5OmniForConditionalGeneration = None
    Qwen2_5OmniProcessor = None
from .utils import format_input_data



# =  =  =  =  =  =  =  =  =  =  =  Logging Setup  =  =  =  =  =  =  =  =  =  =  =  =  =
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =

model_path = "Qwen/Qwen2.5-Omni-7B"

def qwen2_5_omni_7B_model_loader(self):

    if not QWEN_AVAILABLE:
        raise ImportError("Qwen2_5OmniForConditionalGeneration and Qwen2_5OmniProcessor are not available. Please install them with: pip install qwen")

    self.processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
    self.model     = Qwen2_5OmniForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-Omni-7B", torch_dtype=torch.bfloat16, device_map="auto")
    self.model.disable_talker()

    logger.info("Model loaded: {}".format(model_path))



def do_batch_inference(self, audio_arrays, prompts, system_prompts):

    text = [f'<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n<|audio_bos|><|AUDIO|><|audio_eos|>{user_prompt}<|im_end|>\n<|im_start|>assistant\n' for user_prompt, system_prompt in zip(prompts,system_prompts)]

    inputs = self.processor(text=text, audio=audio_arrays, return_tensors="pt", padding=True, use_audio_in_video=True).to(self.model.device).to(self.model.dtype)

    text_ids = self.model.generate(**inputs, return_audio=False)
    outputs = self.processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    responses=[response.split("\nassistant\n")[-1] for response in outputs]

    return responses



def qwen2_5_omni_7B_model_generation(self, input_data):

    audio_array     = input_data["audio"]["array"]
    sampling_rate   = input_data["audio"]["sampling_rate"]
    task_type       = input_data["task_type"]
    instruction     = input_data["text"]

    # Create conversation format
    conversation = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are a speech recognition model." if "ASR" in task_type else "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
            ],
        },
        {
            "role": "user", 
            "content": [
                {"type": "audio", "audio": audio_array},
                {"type": "text", "text": f"{instruction} Only output transcription and nothing else." if "ASR" in task_type else instruction}
            ],
        },
    ]
    
    # Apply chat template
    text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    
    # Prepare inputs directly with audio
    inputs = self.processor(text=text, audio=audio_array, return_tensors="pt", padding=True, use_audio_in_video=True)
    inputs = inputs.to(self.model.device).to(self.model.dtype)
    
    # Generate response
    # text_ids, audio = self.model.generate(**inputs, use_audio_in_video=True)
    text_ids = self.model.generate(**inputs, use_audio_in_video=True)
    
    # Decode response
    output = self.processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    response = output[0].split("assistant\n")[-1] if "assistant\n" in output[0] else output[0]
    
    return response
