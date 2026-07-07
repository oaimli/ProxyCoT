import os.path
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from data_loading import get_full_texts, get_proxy, load_articles, load_train, load_test


def get_statistics(samples):
    questions = []
    answers = []
    prompts_full_context = []
    prompts_proxy_context = []
    for sample in samples:
        questions.append(sample["question"])
        answers.append(sample["answer"])
        prompts_full_context.append(sample["prompt_full_context"])
        prompts_proxy_context.append(sample["prompt_proxy_context"])
    length_questions = []
    length_answers = []
    length_prompts_full_context = []
    length_prompts_proxy_context = []
    for question, answer, prompt_full_context, prompt_proxy_context in tqdm(zip(questions, answers, prompts_full_context, prompts_proxy_context), total=len(questions)):
        tokenized_prompt_full = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_full_context}],
            tokenize=True,
            add_generation_prompt=True)
        length_prompts_full_context.append(len(tokenized_prompt_full))
        tokenized_prompt_proxy = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_proxy_context}],
            tokenize=True,
            add_generation_prompt=True)
        length_prompts_proxy_context.append(len(tokenized_prompt_proxy))
        tokenized_question = tokenizer.encode(question)
        length_questions.append(len(tokenized_question))
        tokenized_answer = tokenizer.encode(answer)
        length_answers.append(len(tokenized_answer))
    print("prompt_full_context", np.mean(length_prompts_full_context))
    print("prompt_proxy_context", np.mean(length_prompts_proxy_context))
    print("question", np.mean(length_questions))
    print("answer", np.mean(length_answers))


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
    # more details in https://arxiv.org/abs/2509.21028
    data_path = "../../../MDRL/bench"

    # get samples with article ids
    train_data, dev_data = load_train(prefixes=["64k", "128k"], samples_folder=data_path + "/dataset/samples/final/", no_null=True, no_simple=True)
    print(len(train_data), len(dev_data))
    test_data_64 = load_test(prefix="64k", samples_folder=data_path + "/dataset/samples/final/", no_null=True, no_simple=True)
    test_data_128 = load_test(prefix="128k", samples_folder=data_path + "/dataset/samples/final/", no_null=True, no_simple=True)
    print(len(train_data), len(dev_data), len(test_data_64), len(test_data_128))

    # get full-text articles
    articles_all = load_articles(articles_folder=data_path + "/article/")

    # some statistics
    questions = set([])
    templates = set([])
    sql_queries = set([])
    for sample in train_data:
        questions.add(sample["question"])
        templates.add(sample["template"])
        sql_queries.add(sample["sql"])
    print(len(questions), len(templates), len(sql_queries))

    target_samples = train_data + dev_data + test_data_64 + test_data_128
    instruction_full = open(os.path.join("../../instructions/qwen_instruction_full_scitrek.txt")).read()
    instruction_metadata = open(os.path.join("../../instructions/qwen_instruction_proxy_scitrek.txt")).read()
    samples_full = []
    for sample_index, sample in enumerate(tqdm(target_samples)):
        question = sample["question"]

        markdowns = get_full_texts(sample, articles_all)
        articles = "\n\n\n".join(markdowns)
        prompt_content_full = instruction_full
        prompt_content_full = prompt_content_full.replace("<question>", question)
        prompt_content_full = prompt_content_full.replace("<articles>", articles)
        sample["prompt_full_context"] = prompt_content_full

        metadata = get_proxy(sample, articles_all)
        prompt_content_proxy = instruction_metadata
        prompt_content_proxy = prompt_content_proxy.replace("<question>", question)
        prompt_content_proxy = prompt_content_proxy.replace("<articles>", metadata)
        sample["prompt_proxy_context"] = prompt_content_proxy

        target_samples[sample_index] = sample

    get_statistics(target_samples)