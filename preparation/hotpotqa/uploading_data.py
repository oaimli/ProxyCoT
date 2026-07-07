from data_loading import load_train, load_test, load_dev, get_full_texts, get_proxy
from datasets import load_dataset
import os
import jsonlines


if __name__ == "__main__":
    # download extended_hotpotqa first from https://drive.google.com/drive/folders/13Ia7sQapLQbBoBWizAMLrVdxsC2TxtIQ?usp=sharing
    dataset_dir = "extended_hotpotqa"
    samples_train, sample_contexts_train, context_texts_train = load_train(data_folder=dataset_dir)
    samples_dev, sample_contexts_dev, context_texts_dev = load_dev(data_folder=dataset_dir)
    samples_test, sample_contexts_test, context_texts_test = load_test(data_folder=dataset_dir)
    print("samples loaded", len(samples_train), len(samples_dev), len(samples_test))


    def process_data(samples, sample_contexts, context_texts):
        tmps = []
        for sample in samples:
            question = sample["question"]
            answer = sample["answer"]
            articles = get_full_texts(sample, sample_contexts, context_texts)
            proxy = get_proxy(sample)
            instruction_full = open("../../instructions/instruction_full_hotpotqa.txt").read()
            instruction_proxy = open("../../instructions/instruction_proxy_hotpotqa.txt").read()
            tmp = {"question": question, "answer": answer, "articles": articles, "instruction_full": instruction_full, "metadata": proxy, "instruction_proxy": instruction_proxy}
            tmps.append(tmp)
        return tmps


    train_data = process_data(samples_train, sample_contexts_train, context_texts_train)
    dev_data = process_data(samples_dev, sample_contexts_dev, context_texts_dev)
    test_data = process_data(samples_test, sample_contexts_test, context_texts_test)
    print("training/dev/test", len(train_data), len(dev_data), len(test_data))

    data_folder = "hotpotqa_data"
    # if not os.path.exists(data_folder):
    #     os.mkdir(data_folder)
    # with jsonlines.open(f"{data_folder}/train.jsonl", "w") as writer:
    #     writer.write_all(train_data)
    # with jsonlines.open(f"{data_folder}/val.jsonl", "w") as writer:
    #     writer.write_all(dev_data)
    # with jsonlines.open(f"{data_folder}/test.jsonl", "w") as writer:
    #     writer.write_all(test_data)

    dataset = load_dataset("json", data_files={"train": f"{data_folder}/train.jsonl", "val": f"{data_folder}/val.jsonl", "test": f"{data_folder}/test.jsonl"})
    dataset.push_to_hub("oaimli/proxycot-hotpotqa")


