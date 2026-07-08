import jsonlines
import os
import random
import sys
from tqdm import tqdm
from transformers import AutoTokenizer
sys.path.append("../../")
from preparation.scitrek.data_loading import load_train, load_articles


def get_context(sample, articles_all):
    cluster_papers = []
    for paper_id in sample["articles"]:
        paper = articles_all[paper_id]
        paper["article_id"] = paper_id
        cluster_papers.append(paper)

    articles = []
    validity = True
    for paper in cluster_papers:
        article = []
        article_title = paper["title"]
        article.append(f"Article title:\n{article_title}")
        authors = paper["authors"]
        article.append(f"Authors:\n{authors}")
        markdown = paper["markdown"].lower()
        reference_index = markdown.rfind("reference")
        appendix_index = markdown.rfind("appendix")
        if appendix_index == -1:
            if reference_index != -1:
                reference_section = markdown[reference_index:]
            else:
                reference_section = ""
        else:
            if reference_index != -1:
                if appendix_index > reference_index:
                    reference_section = markdown[reference_index:appendix_index]
                else:
                    reference_section = markdown[reference_index:]
            else:
                reference_section = ""
        if reference_section == "":
            validity = False
            break
        else:
            article.append(f"Reference:\n{reference_section}")

        articles.append("\n\n".join(article))

    return articles, validity


def process_data(samples, articles_all):
    max_length = 0
    message_data = []
    for sample in tqdm(samples):
        question = sample["question"]
        answer = sample["answer"]
        fields_articles, validity = get_context(sample, articles_all)
        if validity:
            instruction = open("../../instructions/instruction_proxy_scitrek.txt").read()
            instruction = instruction.replace("<question>", question)
            instruction = instruction.replace("<articles>", "\n\n\n".join(fields_articles))
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
    train_message_data = process_data(samples_train, articles_all)
    dev_message_data = process_data(samples_dev, articles_all)
    dev_message_data = random.sample(dev_message_data, 200)
    print("training/dev", len(train_message_data), len(dev_message_data)) # 7290, 100

    saved_folder = "scitrek_data_fields"
    if not os.path.exists(saved_folder):
        os.mkdir(saved_folder)
    with jsonlines.open(f"{saved_folder}/train.jsonl", "w") as writer:
        writer.write_all(train_message_data)
    with jsonlines.open(f"{saved_folder}/val.jsonl", "w") as writer:
        writer.write_all(dev_message_data)
