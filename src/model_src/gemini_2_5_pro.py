#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import re

# add parent directory to sys.path
import sys
sys.path.append('.')
sys.path.append('../')
import logging
import numpy as np
import torch

from tqdm import tqdm

import pathlib
import soundfile as sf

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None

import tempfile


# =  =  =  =  =  =  =  =  =  =  =  Logging Setup  =  =  =  =  =  =  =  =  =  =  =  =  =
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =


def gemini_2_5_pro_model_loader(self):
    """Load Gemini 2.5 Pro model with API key configuration"""
    
    if not GEMINI_AVAILABLE:
        raise ImportError("google-generativeai package is not installed. Please install it with: pip install google-generativeai")
    
    # Configure API key
    if os.getenv("GEMINI_API_KEY"):
        api_key = os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=api_key)
        logger.info("Using GEMINI_API_KEY from environment variable")
    else:
        raise ValueError("No GEMINI_API_KEY found. Please set the environment variable.")
    
    # Initialize Gemini 2.5 Pro model
    self.model_name = "gemini-2.5-pro"
    self.model = genai.GenerativeModel('models/gemini-2.5-pro')
    
    logger.info("Gemini 2.5 Pro model loaded successfully")



def do_sample_inference(self, audio_array, instruction, sampling_rate=16000):
    """Perform inference with Gemini 2.5 Pro model on audio input"""
    
    if not GEMINI_AVAILABLE:
        return "Error: google-generativeai package is not installed"
    
    try:
        # Create temporary audio file
        audio_path = tempfile.NamedTemporaryFile(suffix=".wav", prefix="audio_", delete=False)
        sf.write(audio_path.name, audio_array, sampling_rate)
        
        # Generate content with Gemini 2.5 Pro
        response = self.model.generate_content([
            instruction,
            {
                "mime_type": "audio/wav",
                "data": pathlib.Path(audio_path.name).read_bytes()
            }
        ])
        
        # Clean up temporary file
        os.unlink(audio_path.name)
        
        if response and response.text:
            return response.text.strip()
        else:
            logger.warning("Empty response from Gemini 2.5 Pro")
            return "Error: Empty response"
            
    except Exception as e:
        logger.error(f"Gemini 2.5 Pro API error: {e}")
        # Clean up temporary file if it exists
        if 'audio_path' in locals():
            try:
                os.unlink(audio_path.name)
            except:
                pass
        return f"Error: {str(e)}"



def gemini_2_5_pro_model_generation(self, input):

    audio_array    = input["audio"]["array"]
    sampling_rate  = input["audio"]["sampling_rate"]
    audio_duration = len(audio_array) / sampling_rate
    instruction    = input["text"]

    os.makedirs('tmp', exist_ok=True)

    # For ASR task, if audio duration is more than 30 seconds, we will chunk and infer separately
    if audio_duration > 30 and input['task_type'] == 'ASR':
        logger.info('Audio duration is more than 30 seconds. Chunking and inferring separately.')
        audio_chunks = []
        for i in range(0, len(audio_array), 30 * sampling_rate):
            audio_chunks.append(audio_array[i:i + 30 * sampling_rate])
        
        model_predictions = [do_sample_inference(self, chunk_array, instruction) for chunk_array in tqdm(audio_chunks)]
        output = ' '.join(model_predictions)


    elif audio_duration > 30:
        logger.info('Audio duration is more than 30 seconds. Taking first 30 seconds.')

        audio_array = audio_array[:30 * sampling_rate]
        output = do_sample_inference(self, audio_array, instruction)
    
    else: 
        if audio_duration < 1:
            logger.info('Audio duration is less than 1 second. Padding the audio to 1 second.')
            audio_array = np.pad(audio_array, (0, sampling_rate), 'constant')

        output = do_sample_inference(self, audio_array, instruction)

    return output
