# ProxyCoT
Long-Context Reasoning Through Proxy-Based Chain-of-Thought Tuning, ACL 2026

To use our dataset, please follow the code below.

```python
train_samples = load_dataset("oaimli/proxycot-scitrek", split="train")
dev_samples = load_dataset("oaimli/proxycot-scitrek", split="val")
test_samples = load_dataset("oaimli/proxycot-scitrek", split="test")

for sample in train_samples:
    question = sample["question"]
    answer = sample["answer"]
    metadata = sample["metadata"]
    articles = sample["articles"]
    instruction_proxy = sample["instruction_proxy"]
    instruction_full = sample["instruction_full"]

    instruction_proxy = instruction_proxy.replace("<question>", question)
    instruction_proxy = instruction_proxy.replace("<articles>", "\n\n\n".join(metadata))
    conversation_proxy = [{"role": "user", "content": instruction_proxy}]

    instruction_full = instruction_full.replace("<question>", question)
    instruction_full = instruction_full.replace("<articles>", "\n\n\n".join(articles))
    conversation_full = [{"role": "user", "content": instruction_full}]
```

To use our dataset, please follow the code below.

```python
train_samples = load_dataset("oaimli/proxycot-hotpotqa", split="train")
dev_samples = load_dataset("oaimli/proxycot-hotpotqa", split="val")
test_samples = load_dataset("oaimli/proxycot-hotpotqa", split="test")

for sample in train_samples:
    question = sample["question"]
    answer = sample["answer"]
    metadata = sample["metadata"]
    articles = sample["articles"]
    instruction_proxy = sample["instruction_proxy"]
    instruction_full = sample["instruction_full"]

    instruction_proxy = instruction_proxy.replace("<question>", question)
    instruction_proxy = instruction_proxy.replace("<articles>", "\n\n\n".join(metadata))
    conversation_proxy = [{"role": "user", "content": instruction_proxy}]

    instruction_full = instruction_full.replace("<question>", question)
    instruction_full = instruction_full.replace("<articles>", "\n\n\n".join(articles))
    conversation_full = [{"role": "user", "content": instruction_full}]
```
