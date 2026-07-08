import jsonlines
import os
import random
import sys
import json
sys.path.append("../")

from preparation.scitrek.data_loading import load_test, load_articles


def get_full_texts(sample, papers_all):
    cluster_papers = []
    for paper_id in sample["articles"]:
        paper = papers_all[paper_id]
        cluster_papers.append(paper)

    markdowns = []
    for i, paper in enumerate(cluster_papers):
        markdowns.append(paper["markdown"])

    return markdowns

if __name__ == "__main__":
    dataset_dir = "../../SciTrek/benchmark"
    articles_all = load_articles(articles_folder=dataset_dir + "/article/")

    # read the data from teacher outputs
    samples_test = []
    samples_test.extend(load_test(prefix="64k",
                                  samples_folder=dataset_dir + "/dataset/samples/final/",
                                  no_null=True, no_simple=True))
    samples_test.extend(load_test(prefix="128k",
                                  samples_folder=dataset_dir + "/dataset/samples/final/",
                                  no_null=True, no_simple=True))
    print("original test samples loaded", len(samples_test))

    def process_data(samples):
        message_data = []
        for sample in samples:
            question = sample["question"]
            markdowns = get_full_texts(sample, articles_all)
            articles = "\n\n\n".join(markdowns)
            instruction = open("../instructions/instruction_full_scitrek.txt").read()
            instruction = instruction.replace("<question>", question)
            instruction = instruction.replace("<articles>", articles)

            answer = sample["answer"]
            messages = [{"role": "user", "content": instruction},
                        {"role": "assistant", "content": answer}]
            tmp = {"messages": messages}
            message_data.append(tmp)
        return message_data

    test_message_data = process_data(samples_test)

    data_folder = "scitrek_data_reinforcement_gemma_alex"
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)
    with jsonlines.open(f"{data_folder}/test.jsonl", "w") as writer:
        writer.write_all(test_message_data)