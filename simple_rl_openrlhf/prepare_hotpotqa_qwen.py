import jsonlines
import os
import sys
sys.path.append("../")
from preparation.hotpotqa.data_loading import load_train, load_dev, get_full_texts


if __name__ == "__main__":
    dataset_dir = "../preparation/hotpotqa/extended_hotpotqa"
    samples_train, sample_contexts_train, context_texts_train = load_train(data_folder=dataset_dir)
    samples_dev, sample_contexts_dev, context_texts_dev = load_dev(data_folder=dataset_dir)
    print("samples loaded", len(samples_train), len(samples_dev))

    def process_data(samples, sample_contexts, context_texts):
        message_data = []
        for sample in samples:
            question = sample["question"]
            answer = sample["answer"]
            articles = "\n\n\n".join(get_full_texts(sample, sample_contexts, context_texts))
            instruction = open("../instructions/instruction_full_hotpotqa.txt").read()
            instruction = instruction.replace("<question>", question)
            instruction = instruction.replace("<articles>", articles)
            messages = [{"role": "user", "content": instruction}]
            tmp = {"messages": messages, "answer": answer}
            message_data.append(tmp)
        return message_data

    train_message_data = process_data(samples_train, sample_contexts_train, context_texts_train)
    dev_message_data = process_data(samples_dev, sample_contexts_dev, context_texts_dev)
    print("training/dev", len(train_message_data), len(dev_message_data))

    data_folder = "hotpotqa_data_qwen"
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)
    with jsonlines.open(f"{data_folder}/train.jsonl", "w") as writer:
        writer.write_all(train_message_data)
    with jsonlines.open(f"{data_folder}/val.jsonl", "w") as writer:
        writer.write_all(dev_message_data)
