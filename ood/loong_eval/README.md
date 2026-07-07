# Loong Eval

## Prerequisites
From `loong_eval/`:
```bash
conda create -n torch_vllm python=3.12
conda activate torch_vllm
conda install -y -c nvidia -c conda-forge cuda-toolkit=12.8
conda install -y -c conda-forge curl
pip install --upgrade pip uv
uv pip install --system https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
pip install -r requirements_modern.txt
```
- For GPT-5-mini eval: set your API key in `Loong/config/models/gpt5-mini.yaml`

## 1) Start vLLM Server (only for vLLM-based models)
From `loong_eval/`:
```bash
./start_vllm_server.sh Qwen/Qwen3-30B-A3B-Instruct-2507 8000 /isaparina-loong-pvc/loong_results/qwen3_30b_eval 2
```
- Args: `[model_path] [port] [output_dir] [tensor_parallel_size]`
- Server logs: `<output_dir>/vllm_server.log`
- Stop: `./start_vllm_server.sh stop 8000`

## 2) Run Evaluation
With server running (for vLLM) or directly for API models:
```bash
./run_evaluation.sh <input_jsonl> <output_dir> <eval_model_config>
```
Examples:
- Qwen 3 30B (vLLM):  
  `./run_evaluation.sh /isaparina-loong-pvc/hotpotqa_qwen3_4b_instruct_0527_scitrek_500_test.jsonl /isaparina-loong-pvc/loong_results/qwen3_30b_eval/scitrek_results qwen3_30b.yaml`
- GPT-5-mini (API):  
  `./run_evaluation.sh /path/to/input.jsonl /isaparina-loong-pvc/loong_results/gpt5mini_eval gpt5-mini.yaml`

## 3) Outputs (in `<output_dir>`)
- `converted_for_evaluation.jsonl` — expanded inputs (one per generation)
- `evaluated.jsonl` — per-generation evaluator responses
- `aggregated.jsonl` — averaged scores per original example (used for metrics)

To re-run metrics only:

```bash
cd Loong/src
python step4_cal_metric.py \
  --eval_model qwen3_30b.yaml \
  --evaluate_output_path /isaparina-loong-pvc/loong_results/qwen3_30b_eval/aggregated.jsonl \
  --model_config_dir ../config/models
```