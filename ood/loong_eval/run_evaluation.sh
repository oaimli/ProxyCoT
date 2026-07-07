#!/bin/bash

# Evaluation runner script for Loong evaluation
# Usage: ./run_evaluation.sh [input_file] [output_dir] [eval_model_config]
# Examples:
#   ./run_evaluation.sh input.jsonl output/ qwen3_4b.yaml    # Use Qwen 3 4B (requires vLLM server)
#   ./run_evaluation.sh input.jsonl output/ gpt4.yaml        # Use GPT-4 (requires API key)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Configuration
EVAL_MODEL_CONFIG="${3:-qwen3_4b.yaml}"  # Default: qwen3_4b.yaml, can use gpt4.yaml
MODEL_CONFIG_DIR="${SCRIPT_DIR}/Loong/config/models"
SRC_DIR="${SCRIPT_DIR}/Loong/src"

# Paths
INPUT_FILE="${1:-hotpotqa_qwen3_4b_instruct_0527_zeroshot_test.jsonl}"
OUTPUT_DIR="${2:-/isaparina-loong-pvc/loong_results/qwen3_4b_eval}"
CONVERTED_FILE="${OUTPUT_DIR}/converted_for_evaluation.jsonl"
EVALUATED_FILE="${OUTPUT_DIR}/evaluated.jsonl"
AGGREGATED_FILE="${OUTPUT_DIR}/aggregated.jsonl"

mkdir -p "${OUTPUT_DIR}"

echo "Using evaluator: ${EVAL_MODEL_CONFIG}"
echo ""

echo "INPUT_FILE: ${INPUT_FILE}"

# Step 1: Convert
echo "Step 1: Converting..."
python "${SCRIPT_DIR}/convert_for_evaluation.py" \
    --input_path "${INPUT_FILE}" \
    --output_path "${CONVERTED_FILE}"

# Step 2: Evaluate
echo "Step 2: Evaluating..."
cd "${SRC_DIR}"
python step3_model_evaluate.py \
    --eval_model "${EVAL_MODEL_CONFIG}" \
    --output_path "${CONVERTED_FILE}" \
    --evaluate_output_path "${EVALUATED_FILE}" \
    --process_num_eval 10 \
    --model_config_dir "${MODEL_CONFIG_DIR}"
cd - > /dev/null

# Step 3: Aggregate
echo "Step 3: Aggregating..."
python "${SCRIPT_DIR}/aggregate_scores.py" \
    --input_path "${EVALUATED_FILE}" \
    --output_path "${AGGREGATED_FILE}"

# Step 4: Metrics
echo "Step 4: Calculating metrics..."
cd "${SRC_DIR}"
python step4_cal_metric.py \
    --eval_model "${EVAL_MODEL_CONFIG}" \
    --evaluate_output_path "${AGGREGATED_FILE}" \
    --model_config_dir "${MODEL_CONFIG_DIR}"
cd - > /dev/null

echo "Done! Results in: ${OUTPUT_DIR}"