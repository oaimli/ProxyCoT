from transformers import AutoTokenizer, AutoModelForCausalLM
import huggingface_hub

huggingface_hub.login(token="")

path_hf_weights = "checkpoint_hotpotqa_direct_grounding_qwen/global_step200_hf"
path_hf_tokenizer = "checkpoint_hotpotqa_direct_grounding_qwen/global_step200_hf"
model = AutoModelForCausalLM.from_pretrained(path_hf_weights, torch_dtype="bfloat16")
tokenizer = AutoTokenizer.from_pretrained(path_hf_tokenizer)
print(model.dtype)
model.push_to_hub("oaimli/longtune_hotpotqa_direct_grounding_qwen")
tokenizer.push_to_hub("oaimli/longtune_hotpotqa_direct_grounding_qwen")