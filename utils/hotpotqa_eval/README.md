# HotpotQA LLM Judge Evaluation

## Prerequisites
From `hotpotqa_llm_eval/`:
```bash
pip install openai tqdm numpy
export OPENAI_API_KEY=<YOUR_KEY>
```
We use GPT-5-mini by default.

## Run Evaluation
```bash
python llm_judge_eval.py --input-file <input-file> [--output-file <output-file>] [--max-workers <num_workers>] [--multiple-judges]
```

### Arguments
- `--input-file`: Path to input JSONL file (required)
- `--output-file`: Path to output JSONL file (optional, defaults to `{input_file}_evaluated.jsonl`)
- `--max-workers`: Maximum number of parallel API calls (default: 16)
- `--multiple-judges`: Enable 3 judge evaluations per generation to calculate standard deviation (IGNORE)

### Example
```bash
python llm_judge_eval.py \
  --input-file hotpotqa_proxy_qwen3_235b_a22b_instruct_0527_test.jsonl \
  --output-file hotpotqa_proxy_qwen3_235b_a22b_instruct_0527_test_evaluated.jsonl \
  --max-workers 16
```

## Output Format
The output JSONL file contains all original fields from the input file plus the following evaluation fields:

- `judgments`: List of judgments (True/False) for each generation
  - Single judge mode: `[[True], [False], [True]]` (one judgment per generation)
  - Multiple judges mode: `[[True, True, False], [False, False, False], [True, True, True]]` (3 judgments per generation)
- `judgment_texts`: Raw text responses from the judge API for each judgment
- `accuracies`: List of accuracy scores (0.0-1.0) for each generation
- `accuracy`: Overall accuracy score (mean across all generations)
- `std_devs`: (only with `--multiple-judges`) Standard deviation of judgments for each generation