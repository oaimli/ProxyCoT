import jsonlines
import os
import sys
import random
sys.path.append("../")
from preparation.hotpotqa.data_loading import load_train, load_dev, get_full_texts
from utils.answer_eval import normalize_text, pre_processing


if __name__ == "__main__":
    random.seed(42)
    dataset_dir = "../preparation/hotpotqa/extended_hotpotqa"
    instruction = open("../instructions/instruction_full_hotpotqa.txt").read()
    _, sample_contexts_train, context_texts_train = load_train(data_folder=dataset_dir)
    _, sample_contexts_dev, context_texts_dev = load_dev(data_folder=dataset_dir)

    # read the data from teacher proxy outputs
    samples_train = []
    samples_dev = []
    with jsonlines.open("../reasoning/openrlhf/hotpotqa_proxy_qwen3_4b_instruct_0527_rl_proxy_train.jsonl") as reader:
        for line in reader:
            samples_train.append(line)
    with jsonlines.open("../reasoning/openrlhf/hotpotqa_proxy_qwen3_4b_instruct_0527_rl_proxy_dev.jsonl") as reader:
        for line in reader:
            samples_dev.append(line)
    print("samples loaded", len(samples_train), len(samples_dev))


    def process_data(samples, sample_contexts, context_texts):
        message_data = []
        for sample in samples:
            question = sample["question"]
            articles = "\n\n\n".join(get_full_texts(sample, sample_contexts, context_texts))
            prompt_content = instruction
            prompt_content = prompt_content.replace("<question>", question)
            prompt_content = prompt_content.replace("<articles>", articles)
            reasonings = sample["reasonings"]
            generations = sample["generations"]
            answer = sample["answer"]
            target = ""
            generation_batches, references = pre_processing([generations], [answer])
            for reasoning, generation in zip(reasonings, generation_batches[0]):
                answer = normalize_text(references[0])
                generation = normalize_text(generation)
                if generation == answer:
                    if target == "":
                        target = reasoning
                    else:
                        if len(reasoning) < len(target):
                            target = reasoning

            if target != "":
                messages = [{"role": "user", "content": prompt_content},
                            {"role": "assistant", "content": target}]
                tmp = {"messages": messages}
                message_data.append(tmp)
        return message_data


    train_message_data = process_data(samples_train, sample_contexts_train, context_texts_train)
    dev_message_data = process_data(samples_dev, sample_contexts_dev, context_texts_dev)
    dev_message_data = random.sample(dev_message_data, 200)
    print("training/dev", len(train_message_data), len(dev_message_data))

    data_folder = "hotpotqa_data_reinforcement_qwen_5"
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)
    with jsonlines.open(f"{data_folder}/train.jsonl", "w") as writer:
        writer.write_all(train_message_data)
    with jsonlines.open(f"{data_folder}/dev.jsonl", "w") as writer:
        writer.write_all(dev_message_data)