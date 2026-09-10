#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# SeaLLMs-Audio-7B Model Implementation for AudioBench
# Supports Southeast Asian languages with audio processing
###

import os
import sys
sys.path.append('.')
sys.path.append('../')
import logging
import numpy as np
import torch
from tqdm import tqdm
import librosa
import soundfile as sf
import tempfile

from transformers import AutoProcessor, AutoTokenizer, AutoModel
from transformers import Qwen2AudioForConditionalGeneration
from transformers.generation import GenerationConfig
from src.model_src.qwen2_audio_7b_instruct import post_process_qwen2_asr

# =  =  =  =  =  =  =  =  =  =  =  Logging Setup  =  =  =  =  =  =  =  =  =  =  =  =  =
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =

MODEL_PATH = "SeaLLMs/SeaLLMs-Audio-7B"


def seallms_audio_7b_model_loader(self):
    
    logger.info(f"Loading SeaLLMs-Audio-7B from {MODEL_PATH}")
    
    self.processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True
    )
    self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    ).eval()
    logger.info("SeaLLMs-Audio-7B model loaded successfully")


def seallms_audio_7b_model_generation(self, input):

    audio_array    = input["audio"]["array"]
    sampling_rate  = input["audio"]["sampling_rate"]
    audio_duration = len(audio_array) / sampling_rate

    os.makedirs('tmp', exist_ok=True)

    # For ASR task, if audio duration is more than 30 seconds, we will chunk and infer separately
    if audio_duration > 30 and input['task_type'] == 'ASR':
        logger.info('Audio duration is more than 30 seconds. Chunking and inferring separately.')
        audio_chunks = []
        for i in range(0, len(audio_array), 30 * sampling_rate):
            audio_chunks.append(audio_array[i:i + 30 * sampling_rate])
        
        model_predictions = []
        for chunk in tqdm(audio_chunks):
            audio_path = tempfile.NamedTemporaryFile(suffix=".wav", prefix="audio_", delete=False)
            sf.write(audio_path.name, chunk, sampling_rate)

            conversation = [
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {"role": "user", "content": [
                    {"type": "audio", "audio_url": audio_path.name},
                    {"type": "text", "text": input["text"]},
                ]},
            ]


            text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            audios = []
            for message in conversation:
                if isinstance(message["content"], list):
                    for ele in message["content"]:
                        if ele["type"] == "audio":
                            audios.append(
                                librosa.load(
                                    ele['audio_url'],
                                    sr=self.processor.feature_extractor.sampling_rate)[0]
                            )

            inputs = self.processor(text=text, audios=audios, sampling_rate=self.processor.feature_extractor.sampling_rate, return_tensors="pt", padding=True)
            inputs = inputs.to("cuda")

            generate_ids = self.model.generate(**inputs, max_new_tokens=512)
            generate_ids = generate_ids[:, inputs.input_ids.size(1):]

            response = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

            # Reprocess the results to get the output
            if input['task_type'] in ['ASR', 'ST', 'TCQ']: response = post_process_qwen2_asr(response)

            model_predictions.append(response)

        output = ' '.join(model_predictions)

    elif audio_duration > 30:
        logger.info('Audio duration is more than 30 seconds. Taking first 30 seconds.')
        #audio_path = f"tmp/audio_{self.dataset_name}.wav"
        audio_path = tempfile.NamedTemporaryFile(suffix=".wav", prefix="audio_", delete=False)
        sf.write(audio_path.name, audio_array[:30 * sampling_rate], sampling_rate)

        conversation = [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {"role": "user", "content": [
                {"type": "audio", "audio_url": audio_path.name},
                {"type": "text", "text": input["text"]},
            ]},
        ]

        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios = []
        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if ele["type"] == "audio":
                        audios.append(
                            librosa.load(
                                ele['audio_url'],
                                # BytesIO(urlopen(ele['audio_url']).read()),
                                sr=self.processor.feature_extractor.sampling_rate)[0]
                        )

        inputs = self.processor(text=text, audios=audios, sampling_rate=self.processor.feature_extractor.sampling_rate, return_tensors="pt", padding=True)
        inputs = inputs.to("cuda")

        generate_ids = self.model.generate(**inputs, max_new_tokens=512)
        generate_ids = generate_ids[:, inputs.input_ids.size(1):]

        response = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        output = response

        # Reprocess the results to get the output
        if input['task_type'] in ['ASR', 'ST', 'TCQ']: output = post_process_qwen2_asr(output)


    else:
        if audio_duration < 1:
            logger.info('Audio duration is less than 1 second. Padding the audio to 1 second.')
            audio_array = np.pad(audio_array, (0, sampling_rate), 'constant')

        audio_path = tempfile.NamedTemporaryFile(suffix=".wav", prefix="audio_", delete=False)
        sf.write(audio_path.name, audio_array, sampling_rate)

        conversation = [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {"role": "user", "content": [
                {"type": "audio", "audio_url": audio_path.name},
                {"type": "text", "text": input["text"]},
            ]},
        ]

        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios = []
        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if ele["type"] == "audio":
                        audios.append(
                            librosa.load(
                                ele['audio_url'],
                                sr=self.processor.feature_extractor.sampling_rate)[0]
                        )

        inputs = self.processor(text=text, audios=audios, sampling_rate=self.processor.feature_extractor.sampling_rate, return_tensors="pt", padding=True)
        inputs = inputs.to("cuda")

        generate_ids = self.model.generate(**inputs, max_new_tokens=512)
        generate_ids = generate_ids[:, inputs.input_ids.size(1):]

        response = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        output = response

        # Reprocess the results to get the output
        if input['task_type'] in ['ASR', 'ST', 'TCQ']: output = post_process_qwen2_asr(output)

    return output

