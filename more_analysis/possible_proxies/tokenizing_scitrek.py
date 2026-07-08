import json
import os
import spacy
import sys
from tqdm import tqdm
import multiprocessing
import functools
sys.path.append("../../")
from preparation.scitrek.data_loading import load_articles


def computing_multi_process(i, data):
    sample = data[i]
    article_id = sample["article_id"]
    markdown = sample["markdown"]
    doc = nlp(markdown)
    return {"article_id": article_id, "sentences": [sent.text for sent in doc.sents]}


if __name__ == "__main__":
    nlp = spacy.load("en_core_web_sm")
    dataset_dir = "../../../SciTrek/benchmark"
    articles_all = load_articles(articles_folder=dataset_dir + "/article/")
    articles_all_list = []
    for article_id in tqdm(list(articles_all.keys())):
        article = articles_all[article_id]["markdown"]
        articles_all_list.append({"article_id": article_id, "markdown": article})
    # articles_all_list = articles_all_list[:64]

    partial_computing = functools.partial(computing_multi_process, data=articles_all_list)
    with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
        results = list(tqdm(p.imap(partial_computing, range(len(articles_all_list)), chunksize=16), total=len(articles_all_list),
                            desc="computing_multi_process"))
    sentences_all = {}
    for result in results:
        sentences_all[result["article_id"]] = result["sentences"]

    with open("sentences_all_scitrek_training.json", "w") as f:
        json.dump(sentences_all, f)