# SEA-SpeechBench

Official code for the EMNLP 2026 paper **[SEA-SpeechBench: A Large-Scale Multitask Benchmark for Speech Understanding Across Southeast Asia]**.

- 📄 Paper: <https://arxiv.org/abs/2609.09672>
- 🌐 Project page (datasets, leaderboard, task definitions): <https://zwenyu.github.io/SEA-SpeechBench/>

## Highlights

- **Scale.** 97,194 samples across 99 evaluation sets and 597 hours of curated audio, sourced from 23 public and community corpora.
- **11 SEA languages.** Burmese, English, Filipino, Indonesian, Khmer, Lao, Malay, Mandarin Chinese, Tamil, Thai, Vietnamese — covering the official languages of nine Southeast Asian countries.
- **9 tasks in 3 categories.**
  - *Speech processing* — ASR, ST (Speech Translation), SQA (Spoken QA)
  - *Paralinguistics* — ER (Emotion), GR (Gender), AgeR (Age), SpkR (Speaker Recognition)
  - *Temporal understanding* — TCQ (Timestamped Content Query), TLoc (Temporal Localization); two novel tasks that treat audio as a searchable temporal space, evaluated on clips up to 3 minutes
- **Bilingual prompting.** Every task can be run with either English or native-language instructions (`--use_sea_prompts`), matching how people in the region actually address these systems.
- **Reference evaluation stack.** Model-as-judge (Gemma-3-27B, Qwen3-Omni-30B, …) served via vLLM; open (2B–30B) and commercial (GPT-4o, Gemini 2.5) systems supported out of the box.
- **HPC-ready.** Reference PBS + Enroot job scripts under `hpc_scripts/`.

## Data

Datasets, task specifications and the current leaderboard are hosted on the project page:

<https://zwenyu.github.io/SEA-SpeechBench/>

Download the release and point `DATA_ROOT` at the extracted directory (see [Configuration](#configuration)).

## Installation

```bash
pip install -r requirements.txt
```

Optional dependencies (model-specific): `vllm`, `transformers`, `openai`, `google-generativeai`, `flash-attn`, `librosa`, `soundfile`.

## Configuration

Every path in this repo is referenced through environment variables so nothing is hard-coded. Copy `.env.example` to `.env` and edit to match your machine:

| Variable | Meaning |
|---|---|
| `SPEECHBENCH_REPO_DIR` | Absolute path to this repository |
| `DATA_ROOT`            | Root for evaluation datasets |
| `OUTPUT_ROOT`          | Where inference / evaluation results are written |
| `HF_HOME`              | HuggingFace cache root |
| `HF_DATASETS_CACHE`    | HuggingFace datasets cache |
| `HF_HUB_CACHE`         | HuggingFace hub cache |
| `SCRATCH_ROOT`         | Cluster scratch space |
| `CONTAINER_DIR`        | Directory containing `.sqsh` container images |
| `HF_TOKEN`             | HuggingFace access token (optional) |
| `OPENAI_API_KEY`       | OpenAI key (optional, for GPT-4o) |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI key (optional) |
| `GEMINI_API_KEY`       | Google Gemini key (optional) |

## Running

### Batch (recommended)

Submit every (model × prompt-config × dataset) combination as PBS jobs — one PBS job per (model, `use_sea_prompts`) pair, sharded across GPUs inside the job:

```bash
bash ./hpc_scripts/submit_batch_sea_jobs.sh
```

Edit the `MODEL_NAME_LIST` and `USE_SEA_PROMPTS_LIST` at the top of that script to control what gets submitted.

### Single dataset × model (debug)

```bash
python -m src.main_evaluate_one_dataset \
    --dataset_name asr_gigaspeech2_id_30 \
    --model_name Qwen2-Audio-7B-Instruct \
    --use_sea_prompts True \
    --use_diverse_prompts True \
    --split test_subset1000-short \
    --batch_size 1 \
    --number_of_samples -1 \
    --out_dir $OUTPUT_ROOT
```

### All datasets for one model (multi-GPU)

```bash
python -m src.main_evaluate \
    --model_name Qwen3-Omni-30B-A3B-Thinking \
    --use_sea_prompts True \
    --use_diverse_prompts True \
    --out_dir $OUTPUT_ROOT \
    --vllm_port_list_str 8080n8081
```

## Extending

- **New dataset** — add a loader class under `src/dataset_src/speechbench/` (subclass the appropriate `base_*_dataset.py`) and register it in `src/dataset_src/speechbench/registry.py`.
- **New model** — add a wrapper module under `src/model_src/` exposing `<model_file>_model_loader(self)` and `<model_file>_model_generation(self, input)`, then add an entry to `MODEL_MAP` in `src/model_src/__init__.py`.
- **New metric** — add the metric under `src/metric_src/` and register it in `METRIC_MAP` inside `src/metric_src/__init__.py`. Model-judge prompt templates live in `src/metric_src/prompts/`.

## Citation

If you use SEA-SpeechBench in your research, please cite our paper:

```bibtex
@inproceedings{liao2026seaspeechbench,
  title     = {SEA-SpeechBench: A Large-Scale Multitask Benchmark for Speech Understanding Across Southeast Asia},
  author    = {Liao, Jingyi and Zhang, Wenyu and Liu, Zhuohan and He, Yingxu and Lin, Geyu and Zou, Xunlong and Sun, Shuo and Alsagoff, Syed Ali Redha and Aw, Ai Ti},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year      = {2026}
}
```

## Acknowledgements

Task framing and dataset conventions build on the [AudioBench](https://arxiv.org/abs/2406.16020) evaluation methodology.

## License

See `LICENSE`.
