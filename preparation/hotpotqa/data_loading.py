import random
import jsonlines

random.seed(42)


def load_test(data_folder="extended_hotpotqa"):
    # ['id', 'question', 'answer', 'type', 'level', 'supporting_facts', 'context']
    samples = []
    # ['sample_id', 'context_titles']
    sample_contexts = {}
    # ['title', 'text']
    context_texts = {}

    for length in ["32k", "64k", "96k", "124k"]:
        samples_tmp = []
        with jsonlines.open(f"{data_folder}/test_{length}.jsonl") as reader:
            for line in reader:
                line["length"] = length
                samples_tmp.append(line)
        samples_tmp = random.sample(samples_tmp, 150)
        samples.extend(samples_tmp)

        target_sample_ids = []
        for sample in samples_tmp:
            target_sample_ids.append(sample["id"])

        target_titles = set([])
        with jsonlines.open(f"{data_folder}/test_{length}_example_contexts.jsonl") as reader:
            for line in reader:
                if line['example_id'] in target_sample_ids:
                    context_titles = line["context_titles"]
                    random.shuffle(context_titles)
                    target_titles.update(context_titles)
                    sample_contexts[f"{line['example_id']}_{length}"] = context_titles

        with jsonlines.open(f"{data_folder}/test_{length}_context_texts.jsonl") as reader:
            for line in reader:
                if line["title"] in target_titles:
                    context_texts[line["title"]] = line["text"]

    print(len(samples), len(sample_contexts), len(context_texts))
    return samples, sample_contexts, context_texts


def load_train(data_folder="extended_hotpotqa"):
    # ['id', 'question', 'answer', 'type', 'level', 'supporting_facts', 'context']
    samples = []
    # ['example_id', 'context_titles']
    sample_contexts = {}
    # ['title', 'text']
    context_texts = {}

    for length in ["32k", "64k", "96k", "124k"]:
        with jsonlines.open(f"{data_folder}/train_{length}.jsonl") as reader:
            for line in reader:
                line["length"] = length
                samples.append(line)

        with jsonlines.open(f"{data_folder}/train_{length}_example_contexts.jsonl") as reader:
            for line in reader:
                sample_contexts[f"{line['example_id']}_{length}"] = line["context_titles"]

        with jsonlines.open(f"{data_folder}/train_{length}_context_texts.jsonl") as reader:
            for line in reader:
                context_texts[line["title"]] = line["text"]

    print(len(samples), len(sample_contexts), len(context_texts))
    return samples, sample_contexts, context_texts


def load_dev(data_folder="extended_hotpotqa"):
    # ['id', 'question', 'answer', 'type', 'level', 'supporting_facts', 'context']
    samples = []
    # ['example_id', 'context_titles']
    sample_contexts = {}
    # ['title', 'text']
    context_texts = {}

    for length in ["32k", "64k", "96k", "124k"]:
        samples_tmp = []
        with jsonlines.open(f"{data_folder}/dev_{length}.jsonl") as reader:
            for line in reader:
                line["length"] = length
                samples_tmp.append(line)
        samples_tmp = random.sample(samples_tmp, 100)
        samples.extend(samples_tmp)

        target_sample_ids = []
        for sample in samples_tmp:
            target_sample_ids.append(sample["id"])

        target_titles = set([])
        with jsonlines.open(f"{data_folder}/validation_{length}_example_contexts.jsonl") as reader:
            for line in reader:
                if line['example_id'] in target_sample_ids:
                    context_titles = line["context_titles"]
                    random.shuffle(context_titles)
                    target_titles.update(context_titles)
                    sample_contexts[f"{line['example_id']}_{length}"] = context_titles

        with jsonlines.open(f"{data_folder}/validation_{length}_context_texts.jsonl") as reader:
            for line in reader:
                if line["title"] in target_titles:
                    context_texts[line["title"]] = line["text"]

    print(len(samples), len(sample_contexts), len(context_texts))
    return samples, sample_contexts, context_texts


def get_proxy(sample):
    snippets = []
    target_titles = sample["supporting_facts"]["title"]
    for sentences_batch, title in zip(sample["context"]["sentences"], sample["context"]["title"]):
        if title in target_titles:
            snippets.append(" ".join(sentences_batch))
    return snippets


def get_full_texts(sample, sample_contexts, context_texts):
    sample_id = sample["id"]
    sample_length = sample["length"]
    articles = []
    for title in sample_contexts[f"{sample_id}_{sample_length}"]:
        articles.append(context_texts[title])
    return articles


if __name__ == "__main__":
    samples_test, sample_contexts_test, context_texts_test = load_test()
    samples_train, sample_contexts_train, context_texts_train = load_train()
    samples_dev, sample_contexts_dev, context_texts_dev = load_dev()
