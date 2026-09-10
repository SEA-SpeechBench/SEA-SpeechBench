OUT_ROOT_DIR=${OUTPUT_ROOT}
ROOT_DIR=${DATA_ROOT}/
SPEECHBENCH_REPO_DIR=${ROOT_DIR}/SEA-SpeechBench

MODEL_NAME_LIST=(
  # Qwen3-Omni-30B-A3B-Instruct
  Qwen3-Omni-30B-A3B-Thinking
)

USE_SEA_PROMPTS_LIST=(
  True
  False
)
USE_DIVERSE_PROMPTS=True # some datasets use pre-set prompts, USE_DIVERSE_PROMPTS will be changed to False automatically

for USE_SEA_PROMPTS in "${USE_SEA_PROMPTS_LIST[@]}"; do
  for MODEL_NAME in "${MODEL_NAME_LIST[@]}"; do
    OUT_DIR=${OUT_ROOT_DIR}/${MODEL_NAME}
    echo "OUT_DIR: ${OUT_DIR}"
    qsub -v OUT_DIR=${OUT_DIR},USE_DIVERSE_PROMPTS=${USE_DIVERSE_PROMPTS},USE_SEA_PROMPTS=${USE_SEA_PROMPTS},MODEL_NAME=${MODEL_NAME} ${SPEECHBENCH_REPO_DIR}/hpc_scripts/batch_run_sea_eval.sh
  done
done