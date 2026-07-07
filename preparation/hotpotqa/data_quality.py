import random
import json
from data_loading import load_train, load_dev, load_test, get_proxy

if __name__ == "__main__":
    random.seed(42)
    samples_test, _, _ = load_test()
    samples_train, _, _ = load_train()
    samples_dev, _, _ = load_dev()

    eyeballing = []
    for sample in random.sample(samples_train, 20):
        sentences = get_proxy(sample)
        question = sample["question"]
        answer = sample["answer"]
        eyeballing.append({"question": question, "answer": answer, "sentences": sentences, "label": "train", "validity": 1})
    for sample in random.sample(samples_dev, 20):
        sentences = get_proxy(sample)
        question = sample["question"]
        answer = sample["answer"]
        eyeballing.append({"question": question, "answer": answer, "sentences": sentences, "label": "dev", "validity": 1})
    for sample in random.sample(samples_test, 20):
        sentences = get_proxy(sample)
        question = sample["question"]
        answer = sample["answer"]
        eyeballing.append({"question": question, "answer": answer, "sentences": sentences, "label": "test", "validity": 1})

    with open("eyeballing.json", "w") as f:
        json.dump(eyeballing, f, indent=4)
