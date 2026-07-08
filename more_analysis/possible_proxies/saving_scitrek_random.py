from transformers import AutoTokenizer, AutoModelForCausalLM

# # download the original model
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
# path_hf_weights = "qwen3_tmp"
# path_hf_tokenizer = "qwen3_tmp"
# model.save_pretrained(path_hf_weights, safe_serialization=True)
# tokenizer.save_pretrained(path_hf_tokenizer)

# combine model weights with zero_to_fp32, and save to global_step20_tmp if the checkpoint is global_step20
# python zero_to_fp32.py . global_step20_tmp/
# add the other necessary files from tmp to global_step20_tmp

# load the model with Huggingface
path_hf_weights = "checkpoint_hotpotqa_random/_actor/global_step20_tmp"
path_hf_tokenizer = "checkpoint_hotpotqa_random/_actor/global_step20_tmp"
model = AutoModelForCausalLM.from_pretrained(path_hf_weights, torch_dtype="bfloat16")
tokenizer = AutoTokenizer.from_pretrained(path_hf_tokenizer)
print(model.dtype)
# save the model with safetensors
model.save_pretrained("checkpoint_hotpotqa_random/_actor/global_step20_hf", safe_serialization=True)
tokenizer.save_pretrained("checkpoint_hotpotqa_random/_actor/global_step20_hf")