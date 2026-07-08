import random
import json
import jsonlines
import os
import sys
import bm25s
import Stemmer
from tqdm import tqdm
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer
sys.path.append("../../")
from preparation.hotpotqa.data_loading import load_train, load_dev


def get_metadata(sample):
    sentences = []
    target_titles = sample["supporting_facts"]["title"]
    for sentences_batch, title in zip(sample["context"]["sentences"], sample["context"]["title"]):
        if title in target_titles:
            sentences.extend(sentences_batch)
    return sentences


def get_context(sample, sample_contexts, context_sentences, supporting_sentences):
    sample_id = sample["id"]
    sample_question = sample["question"]
    sample_length = sample["length"]
    sentences_all = []
    for title in sample_contexts[f"{sample_id}_{sample_length}"]:
        sentences_all.extend(context_sentences[title])
    # if len(sentences_all) < 50:
    #     print(f"{sample_id}_{sample_length}")
    # if len(supporting_sentences) > 10:
    #     print(f"{sample_id}_{sample_length}")

    stemmer = Stemmer.Stemmer("english")
    corpus_tokens = bm25s.tokenize(sentences_all, stopwords="en", stemmer=stemmer)
    retriever = bm25s.BM25(corpus=sentences_all)
    retriever.index(corpus_tokens)
    query_tokens = bm25s.tokenize(sample_question, stopwords="en", stemmer=stemmer)
    if len(sentences_all) > 50:
        results_bm25, scores = retriever.retrieve(query_tokens, k=50)
        sentences_bm25 = results_bm25[0]
    else:
        sentences_bm25 = sentences_all
    # print(len(sentences_bm25), sentences_bm25[0])

    sentence_embeddings = []
    response = client.embeddings.create(
        input=sentences_bm25,
        model="text-embedding-3-small"
    )
    for result in response.data:
        sentence_embeddings.append(list(result.embedding))

    response = client.embeddings.create(
        input=sample_question,
        model="text-embedding-3-small"
    )
    question_embedding = list(response.data[0].embedding)

    sentence_similarities = list(cosine_similarity([question_embedding], sentence_embeddings)[0])
    assert len(sentence_similarities) == len(sentences_bm25)

    sentences_ranked = [tuple[1] for tuple in sorted(zip(sentence_similarities, sentences_bm25), reverse=True)]
    assert len(sentences_bm25) == len(sentences_ranked)
    print("QUESTION:", sample_question)
    print("SENTENCE:", [str(tmp) for tmp in sentences_ranked[:2]])
    sentences_related = [str(tmp) for tmp in sentences_ranked[:len(supporting_sentences)]]
    return sentences_related


def process_data(samples, sample_contexts, context_sentences):
    max_length = 0
    message_data = []
    for sample in tqdm(samples):
        question = sample["question"]
        answer = sample["answer"]
        supporting_sentences = get_metadata(sample)
        related_sentences = get_context(sample, sample_contexts, context_sentences, supporting_sentences)
        snippets = "\n\n".join(related_sentences)
        instruction = open("../../instructions/instruction_proxy_hotpotqa.txt").read()
        instruction = instruction.replace("<question>", question)
        instruction = instruction.replace("<snippets>", snippets)
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
    api_key = ""
    client = OpenAI(api_key=api_key)
    random.seed(42)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
    dataset_dir = "../../preparation/hotpotqa/extended_hotpotqa"
    samples_train, sample_context_train, _ = load_train(data_folder=dataset_dir)
    samples_dev, sample_context_dev, _ = load_dev(data_folder=dataset_dir)

    with open("context_sentences_hotpotqa_training.json") as f:
        context_sentences = json.load(f)

    train_message_data = process_data(samples_train, sample_context_train, context_sentences)
    dev_message_data = process_data(samples_dev, sample_context_dev, context_sentences)
    print("training/dev", len(train_message_data), len(dev_message_data))

    saved_folder = "hotpotqa_data_related"
    if not os.path.exists(saved_folder):
        os.mkdir(saved_folder)
    with jsonlines.open(f"{saved_folder}/train.jsonl", "w") as writer:
        writer.write_all(train_message_data)
    with jsonlines.open(f"{saved_folder}/val.jsonl", "w") as writer:
        writer.write_all(dev_message_data)
