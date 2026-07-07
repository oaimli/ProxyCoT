import jsonlines
import numpy as np
from transformers import AutoTokenizer


if __name__ == "__main__":
    model_name_official = "Qwen/Qwen3-4B-Thinking-2507"
    tokenizer = AutoTokenizer.from_pretrained(model_name_official)

    # ['id', 'question', 'answer', 'type', 'level', 'supporting_facts', 'context']
    samples = []
    with jsonlines.open("data/test_124k.jsonl") as reader:
        for line in reader:
            samples.append(line)

    # ['example_id', 'context_titles']
    sample_contexts = {}
    with jsonlines.open("data/test_124k_example_contexts.jsonl") as reader:
        for line in reader:
            sample_contexts[line["example_id"]] = line["context_titles"]

    # ['title', 'text']
    context_texts = {}
    with jsonlines.open("data/test_124k_context_texts.jsonl") as reader:
        for line in reader:
            context_texts[line["title"]] = line["text"]

    proxy_word_counts = []
    long_word_counts = []
    proxy_article_counts = []
    long_article_counts = []
    proxy_articles_zero = 0
    long_articles_zero = 0
    long_token_counts = []
    for sample in samples:
        sample_id = sample["id"]
        print("ground-truth answer", sample["answer"])

        target_titles = sample["supporting_facts"]["title"]
        sentences = []
        for sentences_batch, title in zip(sample["context"]["sentences"], sample["context"]["title"]):
            if title in target_titles:
                sentences.extend(sentences_batch)
        proxy_word_counts.append(len(("\n".join(sentences)).split()))

        articles = []
        for title in sample_contexts[sample_id]:
            articles.append(context_texts[title])
        long_word_counts.append(len(("\n".join(articles)).split()))

        print(f"articles long {len(sample_contexts[sample_id])}", sample_contexts[sample_id])
        print(f"articles proxy {len(target_titles)}", target_titles)

        proxy_article_counts.append(len(target_titles))
        long_article_counts.append(len(sample_contexts[sample_id]))

        if len(target_titles) == 0:
            proxy_articles_zero += 1
        if len(sample_contexts[sample_id]) == 0:
            long_articles_zero += 1

        conversation = [{"role": "user", "content": "\n".join(articles)}]
        tokenized = tokenizer.apply_chat_template(
            conversation,
            tokenize=True,
            )
        long_token_counts.append(len(tokenized))

    print(np.mean(proxy_word_counts), np.max(proxy_word_counts), np.min(proxy_word_counts))
    print(np.mean(long_word_counts), np.max(long_word_counts), np.min(long_word_counts))
    print(np.mean(proxy_article_counts), np.max(proxy_article_counts), np.min(proxy_article_counts))
    print(np.mean(long_article_counts), np.max(long_article_counts), np.min(long_article_counts))
    print(proxy_articles_zero, long_articles_zero)
    print(np.mean(long_token_counts), np.max(long_token_counts), np.min(long_token_counts))