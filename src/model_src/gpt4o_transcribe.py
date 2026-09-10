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

def _audio_to_bytesio(audio_array, sampling_rate=16000):
    """Convert audio array to bytes for OpenAI API"""

    # Normalize audio to [-1, 1] range
    if np.max(np.abs(audio_array)) > 1.0:
        audio_array = audio_array / np.max(np.abs(audio_array))

    # Create a BytesIO buffer
    buffer = io.BytesIO()

    # Write audio to buffer as WAV
    sf.write(buffer, audio_array, sampling_rate, format='WAV')

    # Get the bytes
    buffer.seek(0)
    buffer.name = 'audio.wav'

    return buffer

def gpt4o_transcribe_model_loader(self):
    """Load GPT-4o model with direct audio input capabilities"""

    self.model_name = "gpt-4o-transcribe"
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


def gpt4o_transcribe_model_generation(self, input_data):
    """Generate predictions using GPT-4o with direct audio input"""

    audio_array     = input_data["audio"]["array"]
    sampling_rate   = input_data["audio"]["sampling_rate"]
    task_type       = input_data["task_type"]
    instruction     = input_data["text"]

    assert task_type == "ASR", "GPT-4o Transcribe model can only transcribe audio"

    # Convert audio to bytes
    audio_bytesio = _audio_to_bytesio(audio_array, sampling_rate)

    if audio_bytesio is None:
        logger.error("Failed to convert audio to base64")
        return "Error: Could not process audio"

    completion = self.client.audio.transcriptions.create(
        model=self.model_name,  # Use gpt-4o-transcribe
        file=audio_bytesio,
        temperature=0.0,
        language=input_data["language"],
    )

    response = completion.text

    return response.strip()
