# HPC scripts (PBS + Enroot)

Reference templates for running SEA-SpeechBench on a PBS-managed cluster with Enroot containers.

## Files

| Script | Role |
|---|---|
| `submit_batch_sea_jobs.sh`                    | Top-level launcher. Iterates over `MODEL_NAME_LIST` × `USE_SEA_PROMPTS_LIST` and `qsub`s one job per combination. |
| `batch_run_sea_eval.sh`                       | The PBS job body. Starts the judge vLLM, then runs `python -m src.main_evaluate` inside the container. |
| `start_judge_gemma3-27b-instruct_ngpu.sh`     | Boots a Gemma-3-27B judge server via vLLM on N GPUs and exports `MY_VLLM_PORT_JUDGE` / `VLLM_PORT_LIST_STR`. |

## Before you run

Edit the following in `batch_run_sea_eval.sh` for your site:

- `#PBS -P YOUR_PROJECT_ID`         — your PBS project ID
- `#PBS -l container_image=...`     — path to the built container image (`speechbench.sqsh`)
- `#PBS -l enroot_env_file=...`     — Enroot env file for the container
- `ROOT_DIR`, `SPEECHBENCH_REPO_DIR` — where the repo is checked out
- HF cache paths (`HF_HOME`, `HF_DATASETS_CACHE`, `TRANSFORMERS_CACHE`, `NLTK_DATA`)
- `HF_TOKEN` / `AZURE_OPENAI_API_KEY` / `GEMINI_API_KEY` — leave the `${VAR}` placeholders and export the real values in your job-submission shell (or pull from `.env`)

## Submit

```bash
bash ./hpc_scripts/submit_batch_sea_jobs.sh
```

The launcher writes stdout/stderr to `${OUT_DIR}/logs/`, and per-dataset outputs to `${OUT_DIR}/<split>/<task>/<model>/<dataset>.json` plus a `<dataset>_score.json`.
