import sys
import random
from tqdm import tqdm

sys.path.append("../")
from preparation.scitrek.data_loading import load_test, load_articles, get_proxy, get_full_texts


if __name__ == "__main__":
    dataset_dir = "../../SciTrek/benchmark"
    articles_all = load_articles(articles_folder=dataset_dir + "/article/")
    samples_test = []
    samples_test.extend(load_test(prefix="64k",
                                  samples_folder=dataset_dir + "/dataset/samples/final/",
                                  no_null=True, no_simple=True))
    samples_test.extend(load_test(prefix="128k",
                                  samples_folder=dataset_dir + "/dataset/samples/final/",
                                  no_null=True, no_simple=True))
    print("original test samples loaded", len(samples_test))

    samples_test = random.sample(samples_test, 10)
    for sample_index, sample in tqdm(enumerate(samples_test), total=len(samples_test)):
        question = sample["question"]
        metadata = get_proxy(sample, articles_all)
        articles = get_full_texts(sample, articles_all)
        print("#############question:", question)
        print("metadata:", metadata)
        print("articles:", len(articles))
