#!/bin/bash

# 指定使用的 GPU
export CUDA_VISIBLE_DEVICES=0
export TORCHINDUCTOR_COMPILE_THREADS=10
# 禁用 CUDA Graphs
export VLLM_DISABLE_CUDA_GRAPHS=1

# 记录子进程 PID 的变量
VLLM_PID=""

# 清理函数：杀死 vLLM 进程（如果存在）
cleanup() {
    if [[ -n "$VLLM_PID" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "正在终止 vLLM 服务 (PID: $VLLM_PID)..."
        kill "$VLLM_PID"
        wait "$VLLM_PID" 2>/dev/null
        echo "vLLM 服务已停止。"
    fi
    exit 0
}

# 设置 trap 捕获 SIGINT (Ctrl+C) 和 SIGTERM
trap cleanup SIGINT SIGTERM

# 启动 vLLM 服务（后台运行，以便获取 PID）
echo "运行成功的前提是已经下载了模型在Model文件中..."
python -m vllm.entrypoints.openai.api_server \
    --model  ./MODEL/SQL_Dog_DPO\ 
    --dtype float16 \
    --tensor-parallel-size 1 \
    --port 8192 \
    --max-model-len 16738 \
    --gpu-memory-utilization 0.85 \
    --max-num-batched-tokens 16728 \
    --max-num-seqs 10 \
    
VLLM_PID=$!

echo "vLLM 服务已启动，PID: $VLLM_PID，端口: 8105"
echo "Worker 数量: 8"
echo "按 Ctrl+C 停止服务..."

# 等待 vLLM 进程结束（或被信号中断）
wait "$VLLM_PID"