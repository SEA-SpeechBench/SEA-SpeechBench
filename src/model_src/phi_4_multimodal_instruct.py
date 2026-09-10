
# add parent directory to sys.path
import sys
sys.path.append('.')
sys.path.append('../')
import logging
import numpy as np
import torch
import random
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from .utils import format_input_data


# =  =  =  =  =  =  =  =  =  =  =  Logging Setup  =  =  =  =  =  =  =  =  =  =  =  =  =
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =

model_path = "microsoft/Phi-4-multimodal-instruct"



def phi_4_multimodal_instruct_model_loader(self):

    self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    # Patch the base model class before loading to add missing method
    from transformers import AutoModelForCausalLM
    import types
    
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
        # Handle past_key_values for generation
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        
        # Prepare inputs for the model
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        
        # Add past_key_values if available
        if past_key_values is not None:
            model_inputs["past_key_values"] = past_key_values
            
        # Add any additional kwargs
        model_inputs.update(kwargs)
        
        return model_inputs
    
    # Try to patch the model class before instantiation
    try:
        # Get the model class that will be used
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        model_class = AutoModelForCausalLM._get_model_class(config, trust_remote_code=True)
        
        # Add the method to the class if it doesn't exist
        if not hasattr(model_class, 'prepare_inputs_for_generation'):
            model_class.prepare_inputs_for_generation = prepare_inputs_for_generation
    except Exception as e:
        logger.warning(f"Could not patch model class: {e}")
    
    # Load model according to official documentation
    self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
    
    # Ensure the method exists on the instance
    if not hasattr(self.model, 'prepare_inputs_for_generation'):
        self.model.prepare_inputs_for_generation = types.MethodType(prepare_inputs_for_generation, self.model)
    
    # Try to set generation config, but don't fail if it doesn't exist
    try:
        self.generation_config = GenerationConfig.from_pretrained(model_path, 'generation_config.json')
    except:
        # Create a basic generation config if the file doesn't exist
        self.generation_config = GenerationConfig(
            max_new_tokens=1000,
            num_logits_to_keep=1,
            do_sample=True,
            temperature=0.7
        )
    
    logger.info("Model loaded: {}".format(model_path))




def do_sample_inference(self, audio_array, prompt):

    # Convert audio_array to numpy array if it's not already
    if not isinstance(audio_array, np.ndarray):
        audio_array = np.array(audio_array)
    
    # Ensure audio is in the correct format for Phi-4 processor
    audio = (audio_array, 16000)

    inputs = self.processor(text=prompt, audios=[audio], return_tensors='pt').to('cuda')
    generate_ids = self.model.generate(
            **inputs,
            max_new_tokens=1000,
            num_logits_to_keep=1,
            generation_config=self.generation_config,
        )
    generate_ids = generate_ids[:, inputs['input_ids'].shape[1] :]
    response = self.processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

    return response


def do_batch_inference(self, audio_arrays, prompts):

    # Convert audio arrays to numpy arrays and ensure correct format
    audios = []
    for audio_array in audio_arrays:
        if not isinstance(audio_array, np.ndarray):
            audio_array = np.array(audio_array)
        audios.append((audio_array, 16000))

    inputs       = self.processor(text = prompts, audios=audios, return_tensors='pt').to('cuda')
    generate_ids = self.model.generate(
            **inputs,
            max_new_tokens     = 1000,
            num_logits_to_keep = 1,
            generation_config  = self.generation_config,
        )
    generate_ids = generate_ids[:, inputs['input_ids'].shape[1] :]
    responses = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    return responses



def phi_4_multimodal_instruct_model_generation(self, input_data):

    audio_array     = input_data["audio"]["array"]
    sampling_rate   = input_data["audio"]["sampling_rate"]
    task_type       = input_data["task_type"]
    instruction     = input_data["text"]

    # Create prompt for Phi-4 format
    if "ASR" in task_type:
        prompt = f"<|user|><|audio_1|>{instruction} Only output transcription and nothing else.<|end|><|assistant|>"
    else:
        prompt = f"<|user|><|audio_1|>{instruction}<|end|><|assistant|>"

    # Use the existing do_sample_inference function
    response = do_sample_inference(self, audio_array, prompt)
    
    return response

