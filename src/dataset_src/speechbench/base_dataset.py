import random
import logging
import os
import json
from datasets import Features, Audio, Value, load_from_disk
from abc import ABC, abstractmethod
from src.metric_src.registry import get_metric
from src.dataset_src.speechbench.utils import postprocess_text

class BaseDataset(ABC):
    task_category = None
    task_type = None
    dataset_name = None
    metric_name = None

    def __init__(self, split="test_subset1000", num_sample=-1, num_worker=4, use_diverse_prompts=False, use_sea_prompts=False):
        """
        Initialize the BaseDataset class.

        Args:
            split (str): The split of the dataset.
            num_sample (int): The number of samples to be loaded in the dataset.
            num_worker (int, optional): The number of workers to use for data loading. Defaults to 4.
            use_diverse_prompts (bool, optional): Whether to use diverse prompts. Defaults to False.
            use_sea_prompts (bool, optional): Whether to use SEA prompts. Defaults to False (i.e. English prompts).

        Attributes:
            split (str): Stores the split of the dataset.
            num_sample (int): Stores the number of samples.
            num_worker (int): Stores the number of workers for data loading.
            sampling_rate (int): The sampling rate for audio data, set to 16000 Hz.
            raw_data (datasets.Dataset): The raw data loaded from the dataset.
            language (str): The language of the dataset, extracted from the raw data.
            use_diverse_prompts (bool): Whether to use diverse prompts. Can be overridden in subclasses due to nature of dataset.
            use_sea_prompts (bool): Whether to use SEA prompts.
            all_prompts (dict): A dictionary of prompts, loaded from a JSON file.
        """
        self.split = split
        self.num_sample = num_sample
        self.num_worker = num_worker
        self.sampling_rate = 16000

        raw_data = self._load_raw_data()
        self.raw_data = self._select_samples(raw_data)
        if "language" in self.raw_data[0].keys():
            self.language = self.raw_data[0]["language"]
        elif "language_1" in self.raw_data[0].keys():
            self.language = self.raw_data[0]["language_1"]
        else:
            self.language = None
        self.use_sea_prompts = False if (self.language == "en") else use_sea_prompts
        self.use_diverse_prompts = use_diverse_prompts

        # load prompts
        current_dir = os.path.dirname(__file__)
        prompts_path = os.path.join(current_dir, "../prompts", "sea_prompts.json")
        prompts_path = os.path.abspath(prompts_path)
        if os.path.exists(prompts_path):
            with open(prompts_path, "r", encoding="utf-8") as f:
                self.all_prompts = json.load(f)
        else:
            raise FileNotFoundError(f"sea_prompts.json not found at {prompts_path}")

        self.code2lang = {
            "en": {
                "en": "English",
                "ms": "Bahasa Inggeris",
                "my": "အင်္ဂလိပ်",
                "zh": "英语",
                "th": "อังกฤษ",
                "vi": "Tiếng Anh",
                "id": "Bahasa Inggris",
                "lo": "ພາສາອັງກິດ",
                "km": "អង់គ្លេស",
                "tl": "Ingles",
                "ta": "ஆங்கிலம்",
                "pt": "Inglês"
            },
            "ms": {
                "en": "Malay",
                "ms": "Bahasa Melayu",
                "my": "မလေးစာ",
                "zh": "马来语",
                "th": "ภาษามาเลย์",
                "vi": "Tiếng Mã Lai",
                "id": "Bahasa Melayu",
                "lo": "ພາສາມາເລຍ",
                "km": "ភាសាម៉ាឡា",
                "tl": "Malay",
                "ta": "மலாய்",
                "pt": "Malaio"
            },
            "my": {
                "en": "Burmese",
                "ms": "Bahasa Burma",
                "my": "မြန်မာစာ",
                "zh": "缅甸语",
                "th": "ภาษาพม่า",
                "vi": "Tiếng Miến Điện",
                "id": "Bahasa Burma",
                "lo": "ພາສາມຽນມາ",
                "km": "ភាសាប៊ឺម៉ា",
                "tl": "Burmese",
                "ta": "பர்மீஸ்",
                "pt": "Birmanês"
            },
            "zh": {
                "en": "Chinese",
                "ms": "Bahasa Cina",
                "my": "တရုတ်စာ",
                "zh": "中文",
                "th": "ภาษาจีน",
                "vi": "Tiếng Trung",
                "id": "Bahasa Tionghoa",
                "lo": "ພາສາຈີນ",
                "km": "ភាសាចិន",
                "tl": "Tsino",
                "ta": "சீனம்",
                "pt": "Chinês"
            },
            "th": {
                "en": "Thai",
                "ms": "Bahasa Thai",
                "my": "ထိုင်းစာ",
                "zh": "泰语",
                "th": "ภาษาไทย",
                "vi": "Tiếng Thái",
                "id": "Bahasa Thai",
                "lo": "ພາສາໄທ",
                "km": "ភាសាថៃ",
                "tl": "Thai",
                "ta": "தை",
                "pt": "Tailandês"
            },
            "vi": {
                "en": "Vietnamese",
                "ms": "Bahasa Vietnam",
                "my": "ဗီယက်နမ်စာ",
                "zh": "越南语",
                "th": "ภาษาเวียดนาม",
                "vi": "Tiếng Việt",
                "id": "Bahasa Vietnam",
                "lo": "ພາສາເວັດນາມ",
                "km": "ភាសាវៀតណាម",
                "tl": "Vietnamese",
                "ta": "வியட்நாமீஸ்",
                "pt": "Vietnamita"
            },
            "id": {
                "en": "Indonesian",
                "ms": "Bahasa Indonesia",
                "my": "အင်ဒိုနီးရှားစာ",
                "zh": "印度尼西亚语",
                "th": "ภาษาอินโดนีเซีย",
                "vi": "Tiếng Indonesia",
                "id": "Bahasa Indonesia",
                "lo": "ພາສາອິນໂດເນຊີ",
                "km": "ភាសាឥណ្ឌូណេស៊ី",
                "tl": "Indonesian",
                "ta": "இந்தோனேஷியன்",
                "pt": "Indonésio"
            },
            "lo": {
                "en": "Lao",
                "ms": "Bahasa Lao",
                "my": "လာအိုစာ",
                "zh": "老挝语",
                "th": "ภาษาลาว",
                "vi": "Tiếng Lào",
                "id": "Bahasa Lao",
                "lo": "ພາສາລາວ",
                "km": "ភាសាលាវ",
                "tl": "Lao",
                "ta": "லாவோ",
                "pt": "Lao"
            },
            "km": {
                "en": "Khmer",
                "ms": "Bahasa Khmer",
                "my": "ခမာစာ",
                "zh": "柬埔寨语",
                "th": "ภาษาเขมร",
                "vi": "Tiếng Khmer",
                "id": "Bahasa Khmer",
                "lo": "ພາສາខ្មែរ",
                "km": "ភាសាខ្មែរ",
                "tl": "Khmer",
                "ta": "க்மேர்",
                "pt": "Khmer"
            },
            "tl": {
                "en": "Filipino",
                "ms": "Bahasa Filipina",
                "my": "ဖိလစ်ပိုင်စာ",
                "zh": "菲律宾语",
                "th": "ภาษาฟิลิปปินส์",
                "vi": "Tiếng Filipino",
                "id": "Bahasa Filipina",
                "lo": "ພາສາຟິລິບປິນ",
                "km": "ភាសាហ្វីលីពីន",
                "tl": "Filipino",
                "ta": "பிலிப்பைனோ",
                "pt": "Filipino"
            },
            "ta": {
                "en": "Tamil",
                "ms": "Bahasa Tamil",
                "my": "တမီလ်စာ",
                "zh": "泰米尔语",
                "th": "ภาษาทมิฬ",
                "vi": "Tiếng Tamil",
                "id": "Bahasa Tamil",
                "lo": "ພາສາທາມິນ",
                "km": "ភាសាតាមិល",
                "tl": "Tamil",
                "ta": "தமிழ்",
                "pt": "Tâmil"
            },
            "pt": {
                "en": "Portuguese",
                "ms": "Bahasa Portugis",
                "my": "ပေါ်တူဂီစာ",
                "zh": "葡萄牙语",
                "th": "ภาษาโปรตุเกส",
                "vi": "Tiếng Bồ Đào Nha",
                "id": "Bahasa Portugis",
                "lo": "ພາສາປອກຕຸຍການ",
                "km": "ភាសាប៉័រទុហ្គាល់",
                "tl": "Portuges",
                "ta": "பொர்த்துகீசு",
                "pt": "Português"
            }
        }

    def _load_raw_data(self):
        """
        Loads the raw data from the dataset directory.

        Returns:
            raw_data (datasets.Dataset): The raw data loaded from the dataset.
        """
        raw_data = load_from_disk(f"${DATA_ROOT}/speechbench_datasets/{self.task_category}/{self.split}/{self.dataset_name}")
        return raw_data

    def _select_samples(self, raw_data):
        """
        Select a subset of the raw data.

        If `num_sample` is -1, return the entire raw data.

        If the number of samples requested is more than available samples,
        set `num_sample` to the number of available samples and log an info message.

        Otherwise, shuffle the raw data and select a subset of the first `num_sample` samples.

        Args:
            raw_data (datasets.Dataset): The raw data loaded from the dataset.

        Returns:
            select_data (datasets.Dataset): The selected subset of the raw data.
        """
        if self.num_sample == -1:
            return raw_data

        if len(raw_data) < self.num_sample:
            self.num_sample = len(raw_data)
            logging.info("Number of samples requested is more than available samples. Setting number of samples to {}".format(self.num_sample))

        select_data = raw_data.select(range(self.num_sample))
        return select_data

    def _load_prompts(self):
        """
        Loads prompts for the current task and language.

        Args:
            prompts_filename (str): The filename of the prompts file.

        If `use_diverse_prompts` is False, sets `self.prompts` to None.

        If `use_sea_prompts` is True, loads prompts for the current language.
        Otherwise, loads prompts for English.
        """
        if not self.use_diverse_prompts:
            return None

        prompt_language = self.code2lang.get(self.language)["en"] if self.use_sea_prompts else "English"

        # check that task_type is a valid key in prompts_by_task
        if self.task_type not in self.all_prompts["prompts_by_task"]:
            raise ValueError(f"Invalid task type in sea_prompts.json: {self.task_type}")

        # check that prompt_language is a valid key in prompts_by_task[task_type]
        if prompt_language not in self.all_prompts["prompts_by_task"][self.task_type]:
            raise ValueError(f"Invalid prompt language in sea_prompts.json: {prompt_language}")

        prompts = self.all_prompts["prompts_by_task"][self.task_type][prompt_language]

        return prompts

    def _reformat(self, batch):
        """
        Reformat the batch of dataset to match the required format.

        The required format is:
        {
            "audio": [...],
            "text": [...],
            "answer": [...],
            "task_type": [...],
            "language": [...],
            "audio_duration": [...]
        }

        Args:
            batch: a batch of dataset

        Returns:
            a batch of dataset in the required format
        """
        audio = [x["audio"] for x in batch["context"]]
        if self.use_diverse_prompts:
            # use curated diverse multilingual prompts
            instruction = random.choices(self.prompts, k=len(audio))
        else:
            # use single English prompts saved in the dataset
            assert self.use_sea_prompts == False, "SEA prompts are only available with diverse prompts"
            instruction = [x["text"] for x in batch["instruction"]]

        reference = [x["text"] for x in batch["answer"]]
        language = [x for x in batch["language"]]
        audio_duration = [x for x in batch["audio_length"]]

        return {
            "audio": audio,
            "text": instruction,
            "answer": reference,
            "task_type": [self.task_type] * len(audio),
            "language": language,
            "audio_duration": audio_duration,
        }

    def prepare_model_input(self):
        """
        Prepare model input by reformatting the dataset.

        The dataset is reformatted to have the following columns:
        - audio: the audio data
        - text: the instruction text
        - answer: the reference text
        - task_type: the task type (e.g. ASR, SQA)
        - language: the language (e.g. en, ms)
        - audio_duration: the duration of the audio in seconds

        Args:
            self: the dataset object

        Returns:
            a dataset object with the reformatted columns
        """
        self.prompts = self._load_prompts()

        mapped_data = self.raw_data.map(
            self._reformat,
            num_proc    = self.num_worker,
            batched     = True,
            desc        = "Batch Reformatting data",
        )
        input_data = mapped_data.select_columns(
            [
                "audio",
                "text",
                "answer",
                "task_type",
                "language",
                "audio_duration"
            ]
        )
        input_data = input_data.map(lambda example, idx: {"index": idx}, with_indices=True)

        logging.info(f"\n=  =  =  {self.dataset_name} Sample  =  =  =")
        logging.info(input_data.select(random.sample(range(len(input_data)), 1))[0])
        logging.info("=  =  =  =  =  =  =  =  =  =  =  =\n")

        return input_data

    def format_model_predictions(self, input_data, model_predictions):
        """
        Format model predictions to match the required format.

        The required format is:
        [
            {
                "text": [...],
                "answer": [...],
                "task_type": [...],
                "language": [...],
                "audio_duration": [...],
                "model_prediction": [...]
            }
        ]

        Args:
            input_data: a batch of dataset in the required format
            model_predictions: a list of model predictions

        Returns:
            a list of dictionaries with the model predictions
        """
        data_with_model_predictions = []
        for sample in input_data:
            new_sample = sample.copy()
            del new_sample["audio"]
            new_sample["model_prediction"] = model_predictions.pop(0)
            data_with_model_predictions.append(new_sample)
        return data_with_model_predictions

    def compute_score(self, data_with_model_predictions, vllm_port):
        """
        Computes scores for model predictions using specified evaluation metrics.

        This function processes a batch of data containing model predictions, reference answers,
        and questions. It evaluates the predictions against the references using the specified
        metric, and returns the computed scores.

        Args:
            data_with_model_predictions (list of dict): List of dictionaries where each dictionary
                contains "text" (question), "answer" (reference), and "model_prediction" (model"s prediction).
            vllm_port (int): The port number for the VLLM server.

        Returns:
            dict: A dictionary containing the results of the evaluation. The keys depend on the
                provided metric and may include evaluation scores and additional details.

        Raises:
            ValueError: If an unsupported metric is specified.
        """

        questions   = [item["text"] for item in data_with_model_predictions]
        references  = [item["answer"] for item in data_with_model_predictions]
        predictions = [item["model_prediction"] for item in data_with_model_predictions]

        references = postprocess_text(self, references)
        predictions = postprocess_text(self, predictions)

        metric = get_metric(self.metric_name, vllm_port)
        metric_scores, all_details = metric.evaluate([questions, references, predictions])
        return {"metric_scores": metric_scores, "details": all_details}


