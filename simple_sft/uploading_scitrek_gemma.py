from transformers import AutoTokenizer, AutoModelForCausalLM
import huggingface_hub

huggingface_hub.login(token="")

path_hf_weights = "checkpoint_scitrek_gemma/global_step1125_hf"
path_hf_tokenizer = "checkpoint_scitrek_gemma/global_step1125_hf"
model = AutoModelForCausalLM.from_pretrained(path_hf_weights, torch_dtype="bfloat16")
tokenizer = AutoTokenizer.from_pretrained(path_hf_tokenizer)
print(model.dtype)
model.push_to_hub("oaimli/longtune_scitrek_simple_sft_gemma")
tokenizer.push_to_hub("oaimli/longtune_scitrek_simple_sft_gemma")