import spacy
import json
import sys
from tqdm import tqdm
import multiprocessing
import functools
sys.path.append("../../")
from preparation.hotpotqa.data_loading import load_train, load_dev


def computing_multi_process(i, data):
    sample = data[i]
    title = sample["title"]
    article = sample["article"]
    doc = nlp(article)
    return {"title": title, "sentences": [sent.text for sent in doc.sents]}


if __name__ == '__main__':
    nlp = spacy.load("en_core_web_sm")
    dataset_dir = "../../preparation/hotpotqa/extended_hotpotqa"
    samples_train, sample_contexts_train, context_texts_train = load_train(data_folder=dataset_dir)
    samples_dev, sample_contexts_dev, context_texts_dev = load_dev(data_folder=dataset_dir)
    context_texts = {}
    context_texts.update(context_texts_train)
    context_texts.update(context_texts_dev)
    context_texts_list = []
    for title in tqdm(list(context_texts.keys())):
        article = context_texts[title]
        context_texts_list.append({"title": title, "article": article})

    partial_computing = functools.partial(computing_multi_process, data=context_texts_list)
    with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
        results = list(
            tqdm(p.imap(partial_computing, range(len(context_texts_list)), chunksize=16), total=len(context_texts_list),
                 desc="computing_multi_process"))
    context_sentences = {}
    for result in results:
        context_sentences[result["title"]] = result["sentences"]

    with open("context_sentences_hotpotqa.json", "w") as f:
        json.dump(context_sentences, f)