import random
import json
import jsonlines
import os
import sys
from tqdm import tqdm
from transformers import AutoTokenizer
sys.path.append("../../")
from preparation.hotpotqa.data_loading import load_train, load_dev


def get_metadata(sample):
    sentences = []
    target_titles = sample["supporting_facts"]["title"]
    for sentences_batch, title in zip(sample["context"]["sentences"], sample["context"]["title"]):
        if title in target_titles:
            sentences.extend(sentences_batch)
    return sentences


def get_context(sample, sample_contexts, supporting_sentences, context_sentences):
    sample_id = sample["id"]
    sample_length = sample["length"]
    sentences = []
    for title in sample_contexts[f"{sample_id}_{sample_length}"]:
        sentences.extend(context_sentences[title])
    random_sentences = random.sample(sentences, len(supporting_sentences))
    return random_sentences


def process_data(samples, sample_contexts, context_sentences):
    max_length = 0
    message_data = []
    for sample in tqdm(samples):
        question = sample["question"]
        answer = sample["answer"]
        supporting_sentences = get_metadata(sample)
        random_sentences = get_context(sample, sample_contexts, supporting_sentences, context_sentences)
        snippets = "\n\n".join(random_sentences)
        instruction = open("../../instructions/instruction_proxy_hotpotqa.txt").read()
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


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
    dataset_dir = "../../preparation/hotpotqa/extended_hotpotqa"
    samples_train, sample_contexts_train, context_texts_train = load_train(data_folder=dataset_dir)
    samples_dev, sample_contexts_dev, context_texts_dev = load_dev(data_folder=dataset_dir)
    with open("context_sentences_hotpotqa_training.json") as f:
        context_sentences = json.load(f)

    train_message_data = process_data(samples_train, sample_contexts_train, context_sentences)
    dev_message_data = process_data(samples_dev, sample_contexts_dev, context_sentences)
    print("training/dev", len(train_message_data), len(dev_message_data))

    saved_folder = "hotpotqa_data_random"
    if not os.path.exists(saved_folder):
        os.mkdir(saved_folder)
    with jsonlines.open(f"{saved_folder}/train.jsonl", "w") as writer:
        writer.write_all(train_message_data)
    with jsonlines.open(f"{saved_folder}/val.jsonl", "w") as writer:
        writer.write_all(dev_message_data)
