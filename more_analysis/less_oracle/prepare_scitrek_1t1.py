import jsonlines
import os
import spacy
import random
import sys
import json
from tqdm import tqdm
from transformers import AutoTokenizer
sys.path.append("../../")
from preparation.scitrek.data_loading import load_train, load_articles, get_proxy


def get_context(article_ids, metadata, sentences_all):
    metadata_sentences = [x for x in metadata.split("\n") if x.strip() != "" ]
    # print(len(metadata_sentences))
    sentences = []
    for article_id in article_ids:
        sentences.extend(sentences_all[article_id])
    noise_sentences = random.sample(sentences, len(metadata_sentences) * 1)
    mixed_sentences = noise_sentences + metadata_sentences
    return mixed_sentences


def process_data(samples, articles_all, sentences_all):
    max_length = 0
    message_data = []
    for sample in tqdm(samples):
        question = sample["question"]
        answer = sample["answer"]
        metadata = get_proxy(sample, articles_all)
        article_ids = sample["articles"]
        mixed_sentences = get_context(article_ids, metadata, sentences_all)
        instruction = open("../../instructions/instruction_proxy_scitrek.txt").read()
        instruction = instruction.replace("<question>", question)
        instruction = instruction.replace("<articles>", "\n".join(mixed_sentences))
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
    dataset_dir = "../../../SciTrek/benchmark"
    articles_all = load_articles(articles_folder=dataset_dir + "/article/")
    samples_train, samples_dev = load_train(prefixes=["64k", "128k"], samples_folder=dataset_dir + "/dataset/samples/final/",
                                  no_null=True, no_simple=True)
    with open("../possible_proxies/sentences_all_scitrek.json") as f:
        sentences_all = json.load(f)

    train_message_data = process_data(samples_train, articles_all, sentences_all)
    dev_message_data = process_data(samples_dev, articles_all, sentences_all)
    dev_message_data = random.sample(dev_message_data, 200)
    print("training/dev", len(train_message_data), len(dev_message_data)) # 7290, 100

    saved_folder = "scitrek_data_1t1"
    if not os.path.exists(saved_folder):
        os.mkdir(saved_folder)
    with jsonlines.open(f"{saved_folder}/train.jsonl", "w") as writer:
        writer.write_all(train_message_data)
    with jsonlines.open(f"{saved_folder}/val.jsonl", "w") as writer:
        writer.write_all(dev_message_data)
