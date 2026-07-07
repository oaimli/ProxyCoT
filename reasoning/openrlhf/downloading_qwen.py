from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("oaimli/longtune_scitrek_reasoning_reinforcement_qwen")
tokenizer = AutoTokenizer.from_pretrained("oaimli/longtune_scitrek_reasoning_reinforcement_qwen")
print(model.config.model_type)
print(model.lm_head.weight.shape)