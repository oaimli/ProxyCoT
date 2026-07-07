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
    subset_0 = []
    with jsonlines.open("../teacher_full/hotpotqa_full_qwen3_235b_a22b_thinking_0527_train_0.jsonl") as reader:
        for line in reader:
            subset_0.append(line)
    subset_1 = []
    with jsonlines.open("../teacher_full/hotpotqa_full_qwen3_235b_a22b_thinking_0527_train_1.jsonl") as reader:
        for line in reader:
            subset_1.append(line)
    for sample_0, sample_1 in zip(subset_0, subset_1):
        if "generations" in sample_0.keys():
            samples_train.append(sample_0)
            continue
        if "generations" in sample_1.keys():
            samples_train.append(sample_1)
            continue

    samples_dev = []
    with jsonlines.open("../teacher_full/hotpotqa_full_qwen3_235b_a22b_thinking_0527_dev_1.jsonl") as reader:
        for line in reader:
            if "generations" in line.keys():
                samples_dev.append(line)

    # subset_0 = []
    # with jsonlines.open("../teacher_full/hotpotqa_full_qwen3_235b_a22b_thinking_0527_dev.jsonl") as reader:
    #     for line in reader:
    #         subset_0.append(line)
    # subset_1 = []
    # with jsonlines.open("../teacher_full/hotpotqa_full_qwen3_235b_a22b_thinking_0527_dev_1.jsonl") as reader:
    #     for line in reader:
    #         subset_1.append(line)
    # for sample_0, sample_1 in zip(subset_0, subset_1):
    #     if "generations" in sample_0.keys():
    #         samples_dev.append(sample_0)
    #         continue
    #     if "generations" in sample_1.keys():
    #         samples_dev.append(sample_1)
    #         continue

    print("samples loaded", len(samples_train), len(samples_dev))


    def process_data(samples, sample_contexts, context_texts):
        message_data = []
        for sample in samples:
            # print(sample.keys())
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
    print("training/dev", len(train_message_data), len(dev_message_data))
    dev_message_data = random.sample(dev_message_data, 100)
    print("training/dev", len(train_message_data), len(dev_message_data))

    data_folder = "hotpotqa_data"
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)
    with jsonlines.open(f"{data_folder}/train.jsonl", "w") as writer:
        writer.write_all(train_message_data)
    with jsonlines.open(f"{data_folder}/dev.jsonl", "w") as writer:
        writer.write_all(dev_message_data)