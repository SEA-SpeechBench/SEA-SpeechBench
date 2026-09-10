import re
from typing import List, Dict, Tuple

import numpy as np


def _get_bin_names(bin_edges: List) -> List:
    bin_names = [f"0~{bin_edges[0]}"]

    for i in range(len(bin_edges) - 1):
        bin_names.append(f"{bin_edges[i]}~{bin_edges[i+1]}")

    bin_names.append(f"{bin_edges[-1]}+")

    return bin_names


def format_example(speech_array, instruction_text, task_type, index):
    # convert speech and text into mosaic format so that we can directly use the data collator
    return {
        "context_text"     : None,
        "context_audio"    : speech_array,
        "instruction_text" : instruction_text,
        "instruction_audio": [0],
        "answer_text"      : None,
        "answer_audio"     : [0],
        "task_type"        : task_type,
        "task"             : task_type,
        "index"            : index
    }


def format_input_data(input_data: List, logger, min_chunk_size=1, bin_edges=[30, 120, 300], asr_chunk_size=30, max_chunk_size=30) -> Tuple[List, Dict]:
    """
    For ASR task, if audio duration is more than 30 seconds, we will chunk and infer separately
    If audio duration is less than 1 second, we will pad the audio to 1 second
    For other tasks, if audio duration is more than 30 seconds, we will take first 30 seconds
    """
    bin_names = _get_bin_names(bin_edges)
    formatted_input_data = []
    idx_to_bin_mapper = {}

    for index, input in enumerate(input_data):
        audio_array    = input["audio"]["array"]
        sampling_rate  = input["audio"]["sampling_rate"]
        task_type      = input["task_type"]
        instruction    = input["text"]
        audio_duration = len(audio_array) / sampling_rate
        audio_bin_name = bin_names[np.searchsorted(bin_edges, audio_duration, side='right')]
        idx_to_bin_mapper[index] = audio_bin_name

        if audio_duration > asr_chunk_size and input['task_type'].split("-")[0] == 'ASR':
            logger.info(f'Audio duration is more than {asr_chunk_size} seconds. For ASR task, we will chunk and infer separately.')
            audio_chunks = []
            for i in range(0, len(audio_array), asr_chunk_size * sampling_rate):
                current_chunk = audio_array[i:i + asr_chunk_size * sampling_rate]
                audio_chunks.append(current_chunk)
            
            formatted_input_data.extend([format_example(chunk, instruction, task_type, index) for chunk in audio_chunks])
            
        else:

            if audio_duration < min_chunk_size:
                logger.info(f'Audio duration is less than {min_chunk_size} second. Padding the audio to {min_chunk_size} second.')
                audio_array = np.pad(audio_array, (0, sampling_rate - len(audio_array)), 'constant', constant_values=(0, 0))

            elif audio_duration > max_chunk_size:
                logger.info(f'Audio duration is more than {max_chunk_size} seconds. Taking first {max_chunk_size} seconds.')
                audio_array = audio_array[:max_chunk_size * sampling_rate]
            
            formatted_input_data.append(format_example(audio_array, instruction, task_type, index))
    
    return formatted_input_data, idx_to_bin_mapper



# Mapping for variants that should be normalized.
FILLER_MAPPING = {
    "oh": "oh", 
    "ow": "oh", 
    "ohh": "oh", 
    "ohhh": "oh",
    "um": "um", 
    "umm": "um", 
    "em": "um", 
    "urm": "um",
    "hm": "hm", 
    "hmm": "hm", 
    "mmhmm": "hm", 
    "mm": "hm", 
    "mmm": "hm",
    "err": "err", 
    "er": "err", 
    "eh": "err", 
    "errm": "err",
    "lo": "loh", 
    # "loh": "loh", 
    # "lor": "loh",
    "wa": "wah", 
    "wah": "wah",
    "arrrr": "ah",
    "ar": "ah",
    "arr": "ah",
}

# The complete set of target filler words (after normalization)
TARGET_FILLERS = {"ah", "uh", "oh", "um", "hm", "err", 
                  "lah", "leh", "hah", "huh", "wow", 
                  "wah", "walao", "siah", "mah", "meh", "aiyah"}

def count_leading_trailing_spaces(s):
    leading_spaces = len(s) - len(s.lstrip())
    trailing_spaces = len(s) - len(s.rstrip())
    return leading_spaces, trailing_spaces

def normalize_filler(candidate):
    """
    Strip extra punctuation/whitespace, then split on hyphen.
    For each part, if it is in our mapping or in our target set, 
    we normalize it (using the mapping if available, otherwise lower-case it).
    If any part is not a recognized filler word, return None.
    """
    # Remove common wrapping punctuation and spaces.
    cleaned = candidate.strip(" []()!?,;:").strip()
    if not cleaned:
        return None

    # Split compound filler words connected by hyphens.
    parts = cleaned.split('-')
    normalized_parts = []
    for part in parts:
        lower_part = part.lower()
        if lower_part in FILLER_MAPPING:
            normalized_parts.append(FILLER_MAPPING[lower_part])
        elif lower_part in TARGET_FILLERS:
            normalized_parts.append(lower_part)
        else:
            # If any part is not a filler word candidate, do not change.
            return None
    return "-".join(normalized_parts)

def filler_replacer(match):
    """
    This is the replacement function for re.sub.
    It takes the matched token, cleans and normalizes it.
    If it qualifies as a filler word (or compound filler),
    return the standardized version wrapped in parentheses.
    Otherwise, return the original match.
    """
    token = match.group(0)
    ls, rs = count_leading_trailing_spaces(token)
    norm = normalize_filler(token)
    if norm is not None:
        return " "*ls + f"({norm})" + " "*rs
    else:
        return token

def standardize_fillers(text):
    """
    Process the input text and replace all filler words (even those already
    wrapped or connected with hyphens) with their normalized version 
    wrapped in round brackets.
    
    The regex below uses negative lookbehind and lookahead so that only
    “standalone” sequences (possibly with extra wrapping punctuation) are matched.
    """
    # This pattern matches a sequence that may have leading/trailing punctuation/spaces.
    pattern = r'(?<![a-zA-Z0-9_])([\[\(\s]*[A-Za-z]+(?:-[A-Za-z]+)*[\]\)\?\s]*)(?![a-zA-Z0-9_])'
    text = re.sub(pattern, filler_replacer, text)
    text = re.sub("[\(\[（【]+", "(", text)
    text = re.sub("[\)\]）】]+", ")", text)
    text = re.sub("\s+", " ", text)
    return text.strip()