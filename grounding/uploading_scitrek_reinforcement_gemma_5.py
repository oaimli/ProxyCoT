from transformers import AutoTokenizer, AutoModelForCausalLM
import huggingface_hub

huggingface_hub.login(token="")

path_hf_weights = "saved_tmp/checkpoint_scitrek_reinforcement_gemma_5/global_step475_hf"
path_hf_tokenizer = "saved_tmp/checkpoint_scitrek_reinforcement_gemma_5/global_step475_hf"
model = AutoModelForCausalLM.from_pretrained(path_hf_weights, torch_dtype="bfloat16")
tokenizer = AutoTokenizer.from_pretrained(path_hf_tokenizer)
print(model.dtype)
model.push_to_hub("oaimli/longtune_scitrek_grounding_reinforcement_gemma_5")
tokenizer.push_to_hub("oaimli/longtune_scitrek_grounding_reinforcement_gemma_5")