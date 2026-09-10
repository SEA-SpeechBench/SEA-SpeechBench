# add parent directory to sys.path
import sys
sys.path.append('.')
sys.path.append('../')
import os
import re
import base64
import io
import logging
import numpy as np
import soundfile as sf
from time import sleep

# =  =  =  =  =  =  =  =  =  =  =  Logging Setup  =  =  =  =  =  =  =  =  =  =  =  =  =
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =

from .utils import format_input_data

VLLM_MODEL_NAME = "Qwen/Qwen3-Omni-30B-A3B-Instruct"


def _audio_to_base64(audio_array, sampling_rate=16000):
    if np.max(np.abs(audio_array)) > 1.0:
        audio_array = audio_array / np.max(np.abs(audio_array))
    buf = io.BytesIO()
    sf.write(buf, audio_array, sampling_rate, format="WAV")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _strip_thinking_tags(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _find_vllm_port():
    """Get port from env var, or auto-detect from running vLLM process."""
    port = os.environ.get("MY_VLLM_PORT_JUDGE")
    if port and port != "None":
        return port

    try:
        import subprocess
        result = subprocess.run(
            ["bash", "-c", "ps aux | grep 'vllm serve\\|vllm.entrypoints' | grep -v grep | grep -oP '\\-\\-port \\K[0-9]+'| head -1"],
            capture_output=True, text=True, timeout=5
        )
        detected = result.stdout.strip()
        if detected:
            logger.info(f"Auto-detected vLLM port: {detected}")
            return detected
    except Exception as e:
        logger.warning(f"Port auto-detection failed: {e}")

    port_list = os.environ.get("VLLM_PORT_LIST_STR", "")
    if port_list and port_list != "None":
        first_port = port_list.split("n")[0]
        if first_port.isdigit():
            logger.info(f"Using first port from VLLM_PORT_LIST_STR: {first_port}")
            return first_port

    raise ValueError(
        "Cannot find vLLM port. Make sure the vLLM server is running.\n"
        "Check: ps aux | grep vllm"
    )


def qwen3_omni_30b_a3b_thinking_model_loader(self):
    from openai import OpenAI

    port = _find_vllm_port()
    self.client = OpenAI(base_url=f"http://localhost:{port}/v1", api_key="EMPTY")
    self.vllm_model_name = VLLM_MODEL_NAME
    logger.info(f"Qwen3-Omni-30B-A3B-Thinking using vLLM at port {port}")


def qwen3_omni_30b_a3b_thinking_model_generation(self, input_data):
    audio_array   = input_data["audio"]["array"]
    sampling_rate = input_data["audio"]["sampling_rate"]
    task_type     = input_data["task_type"]
    instruction   = input_data["text"]

    system_text = (
        "You are a speech recognition model." if "ASR" in task_type
        else "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
             "capable of perceiving auditory and visual inputs, as well as generating text and speech."
    )
    user_text = (
        f"{instruction} Only output transcription and nothing else." if "ASR" in task_type
        else instruction
    )

    audio_b64 = _audio_to_base64(audio_array, sampling_rate)

    messages = [
        {"role": "system", "content": system_text},
        {
            "role": "user",
            "content": [
                {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{audio_b64}"}},
                {"type": "text", "text": user_text},
            ],
        },
    ]

    max_retries = 3
    delay = 2
    for attempt in range(max_retries):
        try:
            completion = self.client.chat.completions.create(
                model=self.vllm_model_name,
                messages=messages,
                max_tokens=2048,
                temperature=0.6,
                extra_body={"chat_template_kwargs": {"enable_thinking": True}},
            )
            raw = completion.choices[0].message.content.strip()
            return _strip_thinking_tags(raw)
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"vLLM call failed (attempt {attempt+1}): {e}, retrying in {delay}s")
                sleep(delay)
                delay *= 2
            else:
                logger.error(f"vLLM call failed after {max_retries} attempts: {e}")
                return f"Error: {str(e)}"
