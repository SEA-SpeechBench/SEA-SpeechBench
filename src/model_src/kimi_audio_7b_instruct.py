#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# Created Date: Friday, December 20th 2024, 11:17:41 am
# Author: Assistant
# -----
# Copyright (c) Assistant
# 
# -----
# HISTORY:
# Date&Time 			By	Comments
# ----------			---	----------------------------------------------------------
###

import os
import re
import tempfile
import logging
import numpy as np
import torch
import soundfile as sf
from tqdm import tqdm

# add parent directory to sys.path
import sys
sys.path.append('.')
sys.path.append('../')

# Try to import KimiAudio, fallback if not available
try:
    from kimia_infer.api.kimia import KimiAudio
    KIMI_AVAILABLE = True
except ImportError:
    KIMI_AVAILABLE = False
    print("Warning: KimiAudio not available. Please install kimia-infer package.")

# =  =  =  =  =  =  =  =  =  =  =  Logging Setup  =  =  =  =  =  =  =  =  =  =  =  =  =
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =

model_path = "moonshotai/Kimi-Audio-7B-Instruct"

def kimi_audio_7b_instruct_model_loader(self):
    """
    Load Kimi Audio model
    """
    if not KIMI_AVAILABLE:
        raise ImportError("KimiAudio is not available. Please install kimia-infer package.")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        # Load the model from Hugging Face Hub
        self.model = KimiAudio(model_path=model_path, load_detokenizer=True)
        self.model.to(device)
        logger.info(f"Model loaded: {model_path} on device: {device}")
    except Exception as e:
        logger.error(f"Failed to load model from HF Hub: {e}")
        # Fallback to local path if available
        local_path = "/path/to/your/downloaded/kimia-hf-ckpt"  # Update this path
        if os.path.exists(local_path):
            self.model = KimiAudio(model_path=local_path, load_detokenizer=True)
            self.model.to(device)
            logger.info(f"Model loaded from local path: {local_path}")
        else:
            raise Exception(f"Could not load Kimi Audio model. Error: {e}")
    
    # Set default sampling parameters
    self.sampling_params = {
        "audio_temperature": 0.8,
        "audio_top_k": 10,
        "text_temperature": 0.0,
        "text_top_k": 5,
        "audio_repetition_penalty": 1.0,
        "audio_repetition_window_size": 64,
        "text_repetition_penalty": 1.0,
        "text_repetition_window_size": 16,
    }


def post_process_kimi_output(model_output, task_type):
    """
    Post-process Kimi Audio output based on task type
    """
    if task_type in ['ASR', 'ST', 'TCQ']:
        # For ASR tasks, extract the transcription
        # Remove any extra formatting or metadata
        output = model_output.strip()
        
        # Remove common prefixes/suffixes that might be added
        prefixes_to_remove = [
            "Transcription:",
            "The transcription is:",
            "Here's the transcription:",
            "Text:",
            "The text is:",
        ]
        
        for prefix in prefixes_to_remove:
            if output.lower().startswith(prefix.lower()):
                output = output[len(prefix):].strip()
        
        return output
    else:
        # For other tasks, return as is
        return model_output.strip()


def kimi_audio_7b_instruct_model_generation(self, input_data):
    """
    Generate response using Kimi Audio model
    """
    if not KIMI_AVAILABLE:
        raise ImportError("KimiAudio is not available. Please install kimia-infer package.")
    
    audio_array = input_data["audio"]["array"]
    sampling_rate = input_data["audio"]["sampling_rate"]
    task_type = input_data["task_type"]
    instruction = input_data["text"]
    audio_duration = len(audio_array) / sampling_rate
    
    # Create temporary audio file
    os.makedirs('tmp', exist_ok=True)
    
    # Handle different audio durations
    if audio_duration > 30 and task_type == 'ASR':
        logger.info('Audio duration is more than 30 seconds. Chunking and inferring separately.')
        audio_chunks = []
        for i in range(0, len(audio_array), 30 * sampling_rate):
            audio_chunks.append(audio_array[i:i + 30 * sampling_rate])
        
        model_predictions = []
        for chunk in tqdm(audio_chunks):
            # Save chunk to temporary file
            audio_path = tempfile.NamedTemporaryFile(suffix=".wav", prefix="kimi_audio_", delete=False)
            sf.write(audio_path.name, chunk, sampling_rate)
            
            # Create messages for Kimi Audio
            if task_type == 'ASR':
                messages = [
                    {"role": "user", "message_type": "text", "content": "Please transcribe the following audio:"},
                    {"role": "user", "message_type": "audio", "content": audio_path.name}
                ]
            else:
                messages = [
                    {"role": "user", "message_type": "text", "content": instruction},
                    {"role": "user", "message_type": "audio", "content": audio_path.name}
                ]
            
            try:
                # Generate text output
                _, text_output = self.model.generate(messages, **self.sampling_params, output_type="text")
                processed_output = post_process_kimi_output(text_output, task_type)
                model_predictions.append(processed_output)
            except Exception as e:
                logger.error(f"Error processing audio chunk: {e}")
                model_predictions.append("")
            
            # Clean up temporary file
            os.unlink(audio_path.name)
        
        output = ' '.join(model_predictions)
        
    elif audio_duration > 30:
        logger.info('Audio duration is more than 30 seconds. Taking first 30 seconds.')
        # Truncate audio to 30 seconds
        truncated_audio = audio_array[:30 * sampling_rate]
        audio_path = tempfile.NamedTemporaryFile(suffix=".wav", prefix="kimi_audio_", delete=False)
        sf.write(audio_path.name, truncated_audio, sampling_rate)
        
        # Create messages
        if task_type == 'ASR':
            messages = [
                {"role": "user", "message_type": "text", "content": "Please transcribe the following audio:"},
                {"role": "user", "message_type": "audio", "content": audio_path.name}
            ]
        else:
            messages = [
                {"role": "user", "message_type": "text", "content": instruction},
                {"role": "user", "message_type": "audio", "content": audio_path.name}
            ]
        
        try:
            # Generate text output
            _, text_output = self.model.generate(messages, **self.sampling_params, output_type="text")
            output = post_process_kimi_output(text_output, task_type)
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            output = ""
        
        # Clean up temporary file
        os.unlink(audio_path.name)
        
    else:
        # Handle short audio (less than 30 seconds)
        if audio_duration < 1:
            logger.info('Audio duration is less than 1 second. Padding the audio to 1 second.')
            audio_array = np.pad(audio_array, (0, sampling_rate - len(audio_array)), 'constant', constant_values=(0, 0))
        
        audio_path = tempfile.NamedTemporaryFile(suffix=".wav", prefix="kimi_audio_", delete=False)
        sf.write(audio_path.name, audio_array, sampling_rate)
        
        # Create messages
        if task_type == 'ASR':
            messages = [
                {"role": "user", "message_type": "text", "content": "Please transcribe the following audio:"},
                {"role": "user", "message_type": "audio", "content": audio_path.name}
            ]
        else:
            messages = [
                {"role": "user", "message_type": "text", "content": instruction},
                {"role": "user", "message_type": "audio", "content": audio_path.name}
            ]
        
        try:
            # Generate text output
            _, text_output = self.model.generate(messages, **self.sampling_params, output_type="text")
            output = post_process_kimi_output(text_output, task_type)
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            output = ""
        
        # Clean up temporary file
        os.unlink(audio_path.name)
    
    return output


def kimi_audio_7b_instruct_batch_generation(self, input_data_list):
    """
    Batch generation for multiple audio inputs
    """
    if not KIMI_AVAILABLE:
        raise ImportError("KimiAudio is not available. Please install kimia-infer package.")
    
    outputs = []
    for input_data in tqdm(input_data_list):
        try:
            output = kimi_audio_7b_instruct_model_generation(self, input_data)
            outputs.append(output)
        except Exception as e:
            logger.error(f"Error processing input: {e}")
            outputs.append("")
    
    return outputs
