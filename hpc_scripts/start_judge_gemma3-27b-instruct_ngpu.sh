# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
# Start the VLLM Server as the judge
export TERM=xterm
MIN=5000
MAX=6000

NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "NUM_GPUS: $NUM_GPUS"
NUM_VLLMS=$(( NUM_GPUS < 2 ? NUM_GPUS : 2 ))
echo "NUM_VLLMS: $NUM_VLLMS"
TENSOR_PARALLEL_SIZE=$((NUM_GPUS / NUM_VLLMS))

MY_VLLM_PORT_JUDGE=$(( RANDOM % (MAX - MIN + 1) + MIN ))
export MY_VLLM_PORT_JUDGE
VLLM_PORT_LIST=$(seq $MY_VLLM_PORT_JUDGE $((MY_VLLM_PORT_JUDGE + NUM_VLLMS - 1)))
VLLM_PORT_LIST_STR=$(echo "$VLLM_PORT_LIST" | tr '\n' 'n' | sed 's/n$//')
export VLLM_PORT_LIST_STR
echo "VLLM PORT LIST: $VLLM_PORT_LIST_STR"
echo "MY_VLLM_PORT_JUDGE: $MY_VLLM_PORT_JUDGE"

HF_ENDPOINT=https://hf-mirror.com
HF_HOME=${HF_HOME}
HF_DATASETS_CACHE=${HF_DATASETS_CACHE}
NLTK_DATA=${DATA_ROOT}/cache/nltk_data

enroot create -n vllm ${CONTAINER_DIR}/vllm+vllm-openai+v0.8.2.sqsh

counter=0
for VLLM_PORT in $VLLM_PORT_LIST; do
    GPU_AT_PORT=$(seq $((counter * TENSOR_PARALLEL_SIZE)) $((counter * TENSOR_PARALLEL_SIZE + TENSOR_PARALLEL_SIZE - 1)))
    GPU_AT_PORT_CSV=$(echo "$GPU_AT_PORT" | tr '\n' ',' | sed 's/,$//')
    echo "starting VLLM at Port: $VLLM_PORT for GPU $GPU_AT_PORT_CSV"
    enroot start \
        -r -w \
        -m ${DATA_ROOT}:${DATA_ROOT} \
        -e TERM=xterm \
        -e CUDA_VISIBLE_DEVICES="$GPU_AT_PORT_CSV" \
        -e NLTK_DATA=$NLTK_DATA \
        -e HF_HOME=$HF_HOME \
        -e HF_DATASETS_CACHE=$HF_DATASETS_CACHE \
        -e HF_ENDPOINT=$HF_ENDPOINT \
        -e no_proxy=localhost,127.0.0.1,10.104.0.0/21 \
        -e https_proxy=http://10.104.4.124:10104 \
        -e http_proxy=http://10.104.4.124:10104 \
        vllm \
        --model RedHatAI/gemma-3-27b-it-FP8-dynamic \
        --port $VLLM_PORT \
        --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
        --max-model-len 4096 \
        --max-num-seqs 48 \
        --disable-log-requests \
        --disable-log-stats &

    VLLM_PID=$!

    # Wait until port is ready
    until nc -z localhost $VLLM_PORT; do
        sleep 60
    done

    echo "Started server on port $VLLM_PORT for GPU $GPU_AT_PORT_CSV"
    counter=$((counter+1))
done

