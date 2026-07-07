import jsonlines
from transformers import AutoTokenizer
import os

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
data_folder = "loong_data"
samples = []
with jsonlines.open(f"{data_folder}/loong.jsonl") as reader:
    for line in reader:
        samples.append(line)

# ['level', 'set', 'length', 'type', 'language', 'question', 'instruction', 'prompt_template', 'doc', 'answer', 'shuffle_doc', 'id']
samples_paper = []
samples_financial = []
samples_all = []
article_counts = set([])
types = set([])
answer_lengths = []
for sample in samples:
    if sample["language"] == "en" and len(sample["doc"]) >= 2:
        articles = []
        for doc in sample["doc"]:
            if sample["type"] == "paper":
                with open(data_folder + f"/doc/{sample['type']}/{doc}") as f:
                    articles.append(f.read())
            else:
                target_files = []
                for file in os.listdir(data_folder + f"/doc/{sample['type']}"):
                    if doc in file:
                        target_files.append(file)
                if len(target_files) == 1:
                    with open(data_folder + f"/doc/{sample['type']}/{target_files[0]}") as f:
                        articles.append(f.read())

        if len(articles) == len(sample["doc"]):
            messages = [{"role": "user", "content": "\n\n\n".join(articles)}]
            text_tokenized = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True)
            if 32 * 1024 < len(text_tokenized) < 124 * 1024:
                answer = str(sample["answer"])
                answer_lengths.append(len(answer.split()))
                samples_all.append(sample)
                if sample["type"] == "paper":
                    samples_paper.append(sample)
                if sample["type"] == "financial":
                    samples_financial.append(sample)
                article_counts.add(len(articles))

print(len(samples_paper))
print(len(samples_financial))
print(article_counts)
print(max(answer_lengths), min(answer_lengths))
with jsonlines.open(f"valid_loong.jsonl", "w") as writer:
    writer.write_all(samples_all)
