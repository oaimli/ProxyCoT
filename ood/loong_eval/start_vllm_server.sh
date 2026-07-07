#!/bin/bash

# Script to start vLLM server and wait for it to be ready
# Usage: ./start_vllm_server.sh [model_path] [port] [output_dir] [tensor_parallel_size]
# To stop: ./start_vllm_server.sh stop [port]

set -e

# Get script directory for relative paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Configuration
MODEL_PATH="${1:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
VLLM_PORT="${2:-8000}"
OUTPUT_DIR="${3:-/isaparina-loong-pvc/loong_results/qwen3_30b_eval}"
# Tensor parallel size can be passed as 4th arg or via env VLLM_TENSOR_PARALLEL_SIZE (arg takes priority)
VLLM_TENSOR_PARALLEL_SIZE="${4:-${VLLM_TENSOR_PARALLEL_SIZE:-2}}"

# Server configuration
VLLM_SERVED_MODEL_NAME=$(basename "${MODEL_PATH}")
PID_FILE="${OUTPUT_DIR}/vllm_server.pid"
LOG_FILE="${OUTPUT_DIR}/vllm_server.log"

# Handle stop command
if [ "$1" == "stop" ]; then
    PORT_TO_STOP="${2:-8000}"
    if [ -f "$PID_FILE" ]; then
        SERVER_PID=$(cat "$PID_FILE")
        if kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "Stopping vLLM server (PID: $SERVER_PID)..."
            kill $SERVER_PID 2>/dev/null || true
            wait $SERVER_PID 2>/dev/null || true
            rm -f "$PID_FILE"
            echo "vLLM server stopped"
        else
            rm -f "$PID_FILE"
        fi
    fi
    exit 0
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Check if server is already running
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "vLLM server is already running with PID: $OLD_PID"
        echo "To stop: ./start_vllm_server.sh stop $VLLM_PORT"
        exit 0
    else
        rm -f "$PID_FILE"
    fi
fi

# Function to check if vLLM server is ready
wait_for_vllm_server() {
    local port=$1
    local max_attempts=120
    local attempt=0
    
    echo "Waiting for vLLM server to be ready on port ${port}..."
    
    while [ $attempt -lt $max_attempts ]; do
        if curl -s "http://localhost:${port}/health" > /dev/null 2>&1; then
            echo "vLLM server is ready!"
            return 0
        fi
        echo "Attempt $((attempt + 1))/${max_attempts}: waiting..."
        sleep 5
        attempt=$((attempt + 1))
    done
    
    echo "ERROR: vLLM server failed to start within expected time"
    return 1
}

echo "Starting vLLM server..."
echo "Model: ${MODEL_PATH}"
echo "Port: ${VLLM_PORT}"
echo "Log file: ${LOG_FILE}"
echo ""

# Start vLLM server
python -m vllm.entrypoints.openai.api_server \
    --served-model-name "${VLLM_SERVED_MODEL_NAME}" \
    --model "${MODEL_PATH}" \
    --port "${VLLM_PORT}" \
    --tensor-parallel-size "${VLLM_TENSOR_PARALLEL_SIZE}" \
    --trust-remote-code \
    > "${LOG_FILE}" 2>&1 &

SERVER_PID=$!
echo "$SERVER_PID" > "${PID_FILE}"
echo "Server PID: ${SERVER_PID}"

# Wait for server to be ready
sleep 2

if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "ERROR: vLLM server process died. Check logs: ${LOG_FILE}"
    tail -20 "${LOG_FILE}"
    rm -f "$PID_FILE"
    exit 1
fi

if ! wait_for_vllm_server "${VLLM_PORT}"; then
    echo "Failed to start. Check logs: ${LOG_FILE}"
    tail -20 "${LOG_FILE}"
    rm -f "$PID_FILE"
    exit 1
fi

echo ""
echo "vLLM Server is ready!"
echo "API URL: http://localhost:${VLLM_PORT}/v1"
echo "To stop: ./start_vllm_server.sh stop ${VLLM_PORT}"
