import jsonlines
import os
import random
import sys

sys.path.append("../")
from preparation.hotpotqa.data_loading import load_train, load_dev, get_full_texts

if __name__ == "__main__":
    random.seed(42)
    dataset_dir = "../preparation/hotpotqa/extended_hotpotqa"
    instruction = open("../instructions/instruction_full_hotpotqa.txt").read()
    samples_train, sample_contexts_train, context_texts_train = load_train(data_folder=dataset_dir)
    samples_dev, sample_contexts_dev, context_texts_dev = load_dev(data_folder=dataset_dir)
    print("samples loaded", len(samples_train), len(samples_dev))


    def process_data(samples, sample_contexts, context_texts):
        message_data = []
        for sample in samples:
            question = sample["question"]
            answer = sample["answer"]
            articles = "\n\n\n".join(get_full_texts(sample, sample_contexts, context_texts))
            prompt_content = instruction
            prompt_content = prompt_content.replace("<question>", question)
            prompt_content = prompt_content.replace("<articles>", articles)

            prompt_content = prompt_content.replace("Please reason step by step, and put your final answer without any formatting within \\boxed{}.",
                                              "Respond with only the final answer, without additional explanation or formatting.")
            print(prompt_content[-50:])
            messages = [{"role": "user", "content": prompt_content},
                        {"role": "assistant", "content": answer}]
            tmp = {"messages": messages}
            message_data.append(tmp)
        return message_data


    train_message_data = process_data(samples_train, sample_contexts_train, context_texts_train)
    dev_message_data = process_data(samples_dev, sample_contexts_dev, context_texts_dev)
    dev_message_data = random.sample(dev_message_data, 200)
    print("training/dev", len(train_message_data), len(dev_message_data))

    data_folder = "hotpotqa_data_qwen"
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)
    with jsonlines.open(f"{data_folder}/train.jsonl", "w") as writer:
        writer.write_all(train_message_data)
    with jsonlines.open(f"{data_folder}/dev.jsonl", "w") as writer:
        writer.write_all(dev_message_data)
