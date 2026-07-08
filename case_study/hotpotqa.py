import random
from tqdm import tqdm
import sys

sys.path.append("../")
from preparation.hotpotqa.data_loading import load_test, get_proxy, get_full_texts


if __name__ == "__main__":
    dataset_dir = "../preparation/hotpotqa/extended_hotpotqa"
    samples_test, sample_contexts, context_texts = load_test(data_folder=dataset_dir)

    samples_test = random.sample(samples_test, 10)
    for sample_index, sample in tqdm(enumerate(samples_test), total=len(samples_test)):
        question = sample["question"]
        snippets = "\n\n".join(get_proxy(sample))
        articles = get_full_texts(sample, sample_contexts, context_texts)
        print("question:", question)
        print("snippets:", snippets)
        print("articles:", len(articles))
