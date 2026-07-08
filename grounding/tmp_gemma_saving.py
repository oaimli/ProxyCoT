from transformers import AutoTokenizer, AutoModelForCausalLM
import huggingface_hub


model = AutoModelForCausalLM.from_pretrained("./gemma3_4b_it_alex", torch_dtype="bfloat16")
tokenizer = AutoTokenizer.from_pretrained("./gemma3_4b_it_alex", fix_mistral_regex=True)
print(model.dtype)
# model.save_pretrained("gemma3_4b_it_alex", safe_serialization=True)
# tokenizer.save_pretrained("gemma3_4b_it_alex")

huggingface_hub.login(token="")

model.push_to_hub("oaimli/longtune_scitrek_grounding_reinforcement_5_gemma_alex")
tokenizer.push_to_hub("oaimli/longtune_scitrek_grounding_reinforcement_5_gemma_alex")