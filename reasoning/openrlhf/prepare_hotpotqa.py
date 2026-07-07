import jsonlines
import os
import sys
from transformers import AutoTokenizer
sys.path.append("../../")
from preparation.hotpotqa.data_loading import load_train, load_dev, get_proxy


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
    dataset_dir = "../../preparation/hotpotqa/data"
    samples_train, _, _ = load_train(data_folder=dataset_dir)
    samples_dev, _, _ = load_dev(data_folder=dataset_dir)

    def process_data(samples):
        max_length = 0
        message_data = []
        for sample in samples:
            question = sample["question"]
            answer = sample["answer"]
            snippets = "\n\n".join(get_proxy(sample))
            instruction = open("../../instructions/qwen_instruction_proxy_hotpotqa.txt").read()
            instruction = instruction.replace("<question>", question)
            instruction = instruction.replace("<snippets>", snippets)
            messages = [{"role": "user", "content": instruction}]
            text_tokenized = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True)
            if len(text_tokenized) > max_length:
                max_length = len(text_tokenized)
            tmp = {"messages": messages, "answer": answer}
            message_data.append(tmp)
        print("max length", max_length)
        return message_data

    train_message_data = process_data(samples_train)
    dev_message_data = process_data(samples_dev)
    print("training/dev", len(train_message_data), len(dev_message_data))

    if not os.path.exists("hotpotqa_data"):
        os.mkdir("hotpotqa_data")
    with jsonlines.open("hotpotqa_data/train.jsonl", "w") as writer:
        writer.write_all(train_message_data)
    with jsonlines.open("hotpotqa_data/val.jsonl", "w") as writer:
        writer.write_all(dev_message_data)
