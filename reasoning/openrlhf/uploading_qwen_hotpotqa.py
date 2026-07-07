from transformers import AutoTokenizer, AutoModelForCausalLM
import huggingface_hub

huggingface_hub.login(token="")

path_hf_weights = "checkpoint_hotpotqa_qwen/_actor/global_step150_hf"
path_hf_tokenizer = "checkpoint_hotpotqa_qwen/_actor/global_step150_hf"
model = AutoModelForCausalLM.from_pretrained(path_hf_weights, torch_dtype="bfloat16")
tokenizer = AutoTokenizer.from_pretrained(path_hf_tokenizer)
print(model.dtype)
model.push_to_hub("oaimli/longtune_hotpotqa_reasoning_reinforcement_qwen")
tokenizer.push_to_hub("oaimli/longtune_hotpotqa_reasoning_reinforcement_qwen")