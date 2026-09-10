#!/bin/bash
#PBS -N SpeechBench
#PBS -l select=1:ncpus=112:ngpus=2:mem=889gb:container_engine=enroot
#PBS -l walltime=24:00:00
#PBS -q ic103
#PBS -P YOUR_PROJECT_ID
#PBS -l container_image=${CONTAINER_DIR}/speechbench.sqsh
#PBS -l container_name=speechbench
#PBS -l enroot_env_file=${SCRATCH_ROOT}/workspace/multimodal_trainer/scripts/hpc_scripts/enroot_scripts/env.conf
#PBS -o logs/stdout.txt
#PBS -e logs/stderr.txt

ROOT_DIR=${DATA_ROOT}/
SPEECHBENCH_REPO_DIR=${ROOT_DIR}/SEA-SpeechBench

export HF_TOKEN="${HF_TOKEN}"

HF_ENDPOINT=http://hf-mirror.com
HF_DATASETS_CACHE=${HF_DATASETS_CACHE}
HF_HOME=${HF_HOME}
TRANSFORMERS_CACHE=${HF_HOME}
NLTK_DATA=${DATA_ROOT}/cache/nltk_data
HF_DATASETS_OFFLINE=1
HF_HUB_OFFLINE=0
PYARROW_IGNORE_TIMEZONE=1

AZURE_OPENAI_API_KEY="${AZURE_OPENAI_API_KEY}"
GEMINI_API_KEY="${GEMINI_API_KEY}"

EVALUATE_EXP_ID=jy_${MODEL_NAME}_$(date +"%Y%m%d%H%M%S")

echo "Cleaning up any existing vLLM processes..."
pkill -9 -f vllm 2>/dev/null || true
sleep 3
ps aux | grep vllm | grep -v grep | awk '{print $2}' | xargs -r kill -9 2>/dev/null || true
sleep 2

pbs_env=`env | grep -i "PBS_" | cut -d "=" -f 1 | sed '{:q;N;s/\n/ -e /g;t q}'`

mkdir -p /raid/local/containers/enroot-data/${PBS_JOBID}/shm
rm -f /raid/local/containers/enroot-data/${PBS_JOBID}/shm/*

### conduct inference and model judge evaluation in 2 steps

# For Qwen3-Omni models, start vLLM BEFORE step 1 (model uses vLLM for inference)
# if [ "$MODEL_NAME" = "Qwen3-Omni-30B-A3B-Instruct" ] || [ "$MODEL_NAME" = "Qwen3-Omni-30B-A3B-Thinking" ]; then
# 	echo "Starting vLLM server before inference (Qwen3-Omni mode)..."
# 	source $SPEECHBENCH_REPO_DIR/hpc_scripts/start_judge_qwen3_omni_30b_a3b_instruct_ngpu.sh
# 	echo "VLLM PORT LIST: $VLLM_PORT_LIST_STR"
# 	echo "MY_VLLM_PORT_JUDGE: $MY_VLLM_PORT_JUDGE"
# fi

### step 1: conduct model inference

# enroot start \
# 	-m /data:/data -m /scratch:/scratch -m ${DATA_ROOT}:${DATA_ROOT} -m /raid/local/containers/enroot-data/${PBS_JOBID}/shm:/dev/shm \
# 	-e $pbs_env \
# 	-e HF_HOME=$HF_HOME \
# 	-e HF_DATASETS_CACHE=$HF_DATASETS_CACHE \
# 	-e TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE \
# 	-e NLTK_DATA=$NLTK_DATA \
# 	-e HF_ENDPOINT=$HF_ENDPOINT \
# 	-e HF_DATASETS_OFFLINE=$HF_DATASETS_OFFLINE \
# 	-e HF_HUB_OFFLINE=$HF_HUB_OFFLINE \
# 	-e HF_TOKEN=$HF_TOKEN \
# 	-e AZURE_OPENAI_API_KEY=$AZURE_OPENAI_API_KEY \
# 	-e GEMINI_API_KEY=$GEMINI_API_KEY \
# 	-e PYARROW_IGNORE_TIMEZONE=$PYARROW_IGNORE_TIMEZONE \
# 	-e no_proxy=localhost,127.0.0.1,10.104.0.0/21 \
# 	-e https_proxy=http://10.104.4.124:10104 \
# 	-e http_proxy=http://10.104.4.124:10104 \
# 	-e EVALUATE_EXP_ID=$EVALUATE_EXP_ID \
# 	-e MY_VLLM_PORT_JUDGE=$MY_VLLM_PORT_JUDGE \
# 	-e VLLM_PORT_LIST_STR=$VLLM_PORT_LIST_STR \
# 	speechbench \
# 	bash -c "
# 	mkdir -p $OUT_DIR/logs
# 	mkdir -p ${HF_HOME}
# 	mkdir -p ${HF_DATASETS_CACHE}
# 	mkdir -p ${HF_HOME}/modules
# 	chmod -R 755 ${HF_HOME}
# 	cd ${SPEECHBENCH_REPO_DIR}
# 	if [ \"$MODEL_NAME\" = \"Gemma-3N-E4B-IT\" ] || [ \"$MODEL_NAME\" = \"Gemma-3N-E2B-IT\" ] || [ \"$MODEL_NAME\" = \"Gemma-3N-E4B\" ] || [ \"$MODEL_NAME\" = \"Gemma-3N-E2B\" ]; then
# 		echo 'Installing Gemma-3N specific dependencies in dedicated environment...'
# 		mkdir -p ./gemma3n_packages
# 		pip install --target ./gemma3n_packages timm --no-deps || echo 'Warning: Failed to install timm'
# 		export PYTHONPATH=./gemma3n_packages:\$PYTHONPATH
# 		if [ -n \"\$HF_TOKEN\" ]; then
# 			huggingface-cli login --token \$HF_TOKEN || echo 'Warning: Failed to login to Hugging Face'
# 		fi
# 	elif [ \"$MODEL_NAME\" = \"Kimi-Audio-7B-Instruct\" ]; then
# 		echo 'Installing Kimi Audio specific dependencies in dedicated environment...'
# 		mkdir -p ./kimi_packages
# 		pip install --target ./kimi_packages kimia-infer || echo 'Warning: Failed to install kimia-infer'
# 		export PYTHONPATH=./kimi_packages:\$PYTHONPATH
# 	elif [ \"$MODEL_NAME\" = \"Qwen2.5-Omni-3B\" ] || [ \"$MODEL_NAME\" = \"Qwen2.5-Omni-7B\" ]; then
# 		echo 'Installing Qwen2.5-Omni specific dependencies in dedicated environment...'
# 		mkdir -p ./qwen2_5_omni_3B_packages
# 		pip install --target ./qwen2_5_omni_3B_packages git+https://github.com/huggingface/transformers@v4.51.3-Qwen2.5-Omni-preview
# 		export PYTHONPATH=./qwen2_5_omni_3B_packages:\$PYTHONPATH
# 	elif [ \"$MODEL_NAME\" = \"Qwen3-Omni-30B-A3B-Instruct\" ] || [ \"$MODEL_NAME\" = \"Qwen3-Omni-30B-A3B-Thinking\" ]; then
# 		echo 'Installing Qwen3-Omni-30B specific dependencies in dedicated environment...'
# 		mkdir -p ./qwen3_omni_packages
# 		pip install --target ./qwen3_omni_packages git+https://github.com/huggingface/transformers@main || echo 'Warning: Failed to install transformers@main for Qwen3-Omni'
# 		rm -rf ./qwen3_omni_packages/numpy ./qwen3_omni_packages/numpy-*.dist-info
# 		export PYTHONPATH=./qwen3_omni_packages:\$PYTHONPATH
# 	elif [ \"$MODEL_NAME\" = \"gemini-2.5-pro\" ] || [ \"$MODEL_NAME\" = \"gemini-2.5-flash\" ] || [ \"$MODEL_NAME\" = \"gemini-1.5-flash\" ]; then
# 		echo 'Installing Gemini specific dependencies in dedicated environment...'
# 		mkdir -p ./gemini_packages
# 		pip install --target ./gemini_packages google-generativeai || echo 'Warning: Failed to install google-generativeai'
# 		export PYTHONPATH=./gemini_packages:\$PYTHONPATH
# 	fi
# 	export MY_VLLM_PORT_JUDGE=$MY_VLLM_PORT_JUDGE
# 	python -m src.main_evaluate --use_diverse_prompts $USE_DIVERSE_PROMPTS --use_sea_prompts $USE_SEA_PROMPTS --model_name $MODEL_NAME --out_dir $OUT_DIR 2> ${OUT_DIR}/logs/sea_eval_$(date +%Y%m%d%H%M%S).stderr.txt 1> ${OUT_DIR}/logs/sea_eval_$(date +%Y%m%d%H%M%S).stdout.txt
#   "

### step 2: conduct evaluation with model judge

# Start Qwen3-Omni judge server (used by SQA and SpR datasets)
# source $SPEECHBENCH_REPO_DIR/hpc_scripts/start_judge_qwen3_omni_30b_a3b_instruct_ngpu.sh
source $SPEECHBENCH_REPO_DIR/hpc_scripts/start_judge_gemma3-27b-instruct_ngpu.sh
echo "VLLM PORT LIST: $VLLM_PORT_LIST_STR"
echo "MY_VLLM_PORT_JUDGE: $MY_VLLM_PORT_JUDGE"

enroot start \
	-m /data:/data -m /scratch:/scratch -m ${DATA_ROOT}:${DATA_ROOT} -m /raid/local/containers/enroot-data/${PBS_JOBID}/shm:/dev/shm \
	-e $pbs_env \
	-e TERM=xterm \
	-e HF_HOME=$HF_HOME \
	-e HF_DATASETS_CACHE=$HF_DATASETS_CACHE \
	-e NLTK_DATA=$NLTK_DATA \
	-e HF_ENDPOINT=$HF_ENDPOINT \
	-e TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE \
	-e HF_DATASETS_OFFLINE=$HF_DATASETS_OFFLINE \
	-e HF_HUB_OFFLINE=$HF_HUB_OFFLINE \
	-e HF_TOKEN=$HF_TOKEN \
	-e AZURE_OPENAI_API_KEY=$AZURE_OPENAI_API_KEY \
	-e GEMINI_API_KEY=$GEMINI_API_KEY \
	-e PYARROW_IGNORE_TIMEZONE=$PYARROW_IGNORE_TIMEZONE \
	-e no_proxy=localhost,127.0.0.1,10.104.0.0/21 \
	-e https_proxy=http://10.104.4.124:10104 \
	-e http_proxy=http://10.104.4.124:10104 \
	-e EVALUATE_EXP_ID=$EVALUATE_EXP_ID \
	-e VLLM_PORT_LIST_STR=$VLLM_PORT_LIST_STR \
	-e MY_VLLM_PORT_JUDGE=$MY_VLLM_PORT_JUDGE \
	speechbench \
	bash -c "
	mkdir -p $OUT_DIR/logs
	mkdir -p ${HF_HOME}
	mkdir -p ${HF_DATASETS_CACHE}
	mkdir -p ${HF_HOME}/modules
	chmod -R 755 ${HF_HOME}
	cd ${SPEECHBENCH_REPO_DIR}
	if [ \"$MODEL_NAME\" = \"Qwen2.5-Omni-3B\" ] || [ \"$MODEL_NAME\" = \"Qwen2.5-Omni-7B\" ]; then
		echo 'Installing Qwen2.5-Omni specific dependencies...'
		pip install --target ./qwen2_5_omni_3B_packages git+https://github.com/huggingface/transformers@v4.51.3-Qwen2.5-Omni-preview
		export PYTHONPATH=./qwen2_5_omni_3B_packages:\$PYTHONPATH
	elif [ \"$MODEL_NAME\" = \"Qwen3-Omni-30B-A3B-Instruct\" ] || [ \"$MODEL_NAME\" = \"Qwen3-Omni-30B-A3B-Thinking\" ]; then
		echo 'Installing Qwen3-Omni-30B specific dependencies...'
		mkdir -p ./qwen3_omni_packages
		pip install --target ./qwen3_omni_packages git+https://github.com/huggingface/transformers@main || echo 'Warning: Failed to install transformers@main for Qwen3-Omni'
		rm -rf ./qwen3_omni_packages/numpy ./qwen3_omni_packages/numpy-*.dist-info
		export PYTHONPATH=./qwen3_omni_packages:\$PYTHONPATH
	elif [ \"$MODEL_NAME\" = \"gemini-2.5-pro\" ] || [ \"$MODEL_NAME\" = \"gemini-2.5-flash\" ] || [ \"$MODEL_NAME\" = \"gemini-1.5-flash\" ]; then
		echo 'Installing Gemini specific dependencies...'
		pip install --target ./gemini_packages google-generativeai || echo 'Warning: Failed to install google-generativeai'
		export PYTHONPATH=./gemini_packages:\$PYTHONPATH
	fi
	# pip install --upgrade 'numpy==1.24' || echo 'Warning: Failed to upgrade NumPy'
	# pip install --upgrade 'huggingface-hub>=0.30.0,<1.0' || echo 'Warning: Failed to upgrade Hugging Face Hub'
	# Set MY_VLLM_PORT_JUDGE environment variable for the Python process
	export MY_VLLM_PORT_JUDGE=$MY_VLLM_PORT_JUDGE
	python -m src.main_evaluate --vllm_port_list_str $VLLM_PORT_LIST_STR --use_diverse_prompts $USE_DIVERSE_PROMPTS --use_sea_prompts $USE_SEA_PROMPTS --model_name $MODEL_NAME --out_dir $OUT_DIR 2> ${OUT_DIR}/logs/sea_eval_$(date +%Y%m%d%H%M%S).stderr.txt 1> ${OUT_DIR}/logs/sea_eval_$(date +%Y%m%d%H%M%S).stdout.txt
  "

pkill -f vllm

rm -rf /raid/local/containers/enroot-data/${PBS_JOBID}/shm