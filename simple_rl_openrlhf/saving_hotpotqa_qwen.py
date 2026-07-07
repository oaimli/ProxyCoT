from transformers import AutoTokenizer, AutoModelForCausalLM

# # download the original model
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B-Instruct-2507", torch_dtype="bfloat16")
# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
# path_hf_weights = "checkpoint_hotpotqa_qwen/_actor/tmp"
# path_hf_tokenizer = "checkpoint_hotpotqa_qwen/_actor/tmp"
# model.save_pretrained(path_hf_weights, safe_serialization=True)
# tokenizer.save_pretrained(path_hf_tokenizer)

# combine model weights with zero_to_fp32, and save to global_step1790_tmp
# python zero_to_fp32.py . global_step1790_tmp/
# add the other necessary files from tmp to global_step1790_tmp

# load the model with Huggingface
path_hf_weights = "checkpoint_hotpotqa_qwen/_actor/global_step1790_tmp"
path_hf_tokenizer = "checkpoint_hotpotqa_qwen/_actor/global_step1790_tmp"
model = AutoModelForCausalLM.from_pretrained(path_hf_weights, torch_dtype="bfloat16")
tokenizer = AutoTokenizer.from_pretrained(path_hf_tokenizer)
print(model.dtype)
# save the model with safetensors
model.save_pretrained("checkpoint_hotpotqa_qwen/_actor/global_step1790_hf", safe_serialization=True)
tokenizer.save_pretrained("checkpoint_hotpotqa_qwen/_actor/global_step1790_hf")