import os
import sys
import logging
import json
import base64
import io
from typing import List
import numpy as np
import soundfile as sf
from tqdm import tqdm
from time import sleep
from openai import OpenAI, AzureOpenAI

# =  =  =  =  =  =  =  =  =  =  =  Logging Setup  =  =  =  =  =  =  =  =  =  =  =  =  =
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

def _audio_to_base64(audio_array, sampling_rate=16000):
    """Convert audio array to base64 encoded format for OpenAI API"""

    # Normalize audio to [-1, 1] range
    if np.max(np.abs(audio_array)) > 1.0:
        audio_array = audio_array / np.max(np.abs(audio_array))

    # Create a BytesIO buffer
    buffer = io.BytesIO()

    # Write audio to buffer as WAV
    sf.write(buffer, audio_array, sampling_rate, format='WAV')

    # Get the bytes and encode to base64
    audio_bytes = buffer.getvalue()
    audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

    return audio_base64

def gpt4o_audio_preview_model_loader(self):
    """Load GPT-4o model with direct audio input capabilities"""

    self.model_name = "gpt-4o-audio-preview"
    if os.getenv("AZURE_OPENAI_API_KEY"):
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.client = AzureOpenAI(
            azure_endpoint='https://alillm2.openai.azure.com/',
            api_key=api_key,
            api_version="2025-04-01-preview"
        )
        logger.info(f"Using Azure OpenAI API.")
    elif os.getenv("OPENAI_API_KEY"):
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)
        logger.info(f"Using regular OpenAI API.")
    else:
        raise ValueError("No API key found.")

    logger.info("GPT-4o Audio Preview model loaded successfully")


def gpt4o_audio_preview_model_generation(self, input_data):
    """Generate predictions using GPT-4o with direct audio input"""

    audio_array     = input_data["audio"]["array"]
    sampling_rate   = input_data["audio"]["sampling_rate"]
    task_type       = input_data["task_type"]
    instruction     = input_data["text"]

    # Convert audio to base64
    audio_base64 = _audio_to_base64(audio_array, sampling_rate)

    if audio_base64 is None:
        logger.error("Failed to convert audio to base64")
        return "Error: Could not process audio"

    # Create messages with system prompt and audio input
    messages = [
        {
            "role": "system",
            "content": "You are an expert audio analysis system. Analyze the provided audio carefully and provide accurate, specific responses based on the given task."
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": instruction
                },
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": audio_base64,
                        "format": "wav"
                    }
                }
            ]
        }
    ]

    # Add retry mechanism for rate limiting
    max_retries = 3
    retry_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,  # Use gpt-4o-audio-preview
                messages=messages,
                max_tokens=200
            )
            break  # Success, exit retry loop
        except Exception as e:
            if "429" in str(e) or "rate limit" in str(e).lower():
                if attempt < max_retries - 1:
                    logger.warning(f"Rate limit hit, retrying in {retry_delay} seconds... (attempt {attempt + 1}/{max_retries})")
                    sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                else:
                    logger.error(f"Rate limit exceeded after {max_retries} attempts")
                    return "Error: Rate limit exceeded, please try again later"
            else:
                logger.error(f"API error: {e}")
                return f"Error: {str(e)}"

    response = completion.choices[0].message.content

    if response:
        return response.strip()
    else:
        logger.warning("Empty response from GPT-4o")
        return "Error: Empty response"