import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm
from data_loading import load_train, load_dev, load_test, get_proxy, get_full_texts


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

    samples_test, sample_contexts_test, context_texts_test = load_test()
    samples_train, sample_contexts_train, context_texts_train = load_train()
    samples_dev, sample_contexts_dev, context_texts_dev = load_dev()

    instruction_proxy = open("../../instructions/qwen_instruction_proxy_hotpotqa.txt").read()
    instruction_full = open("../../instructions/qwen_instruction_full_hotpotqa.txt").read()

    target_samples = []
    for sample in samples_test:
        sample["label"] = "test"
        target_samples.append(sample)
    for sample in samples_train:
        sample["label"] = "train"
        samples_train.append(sample)
    for sample in samples_dev:
        sample["label"] = "dev"
        target_samples.append(sample)
    print("test samples", len(samples_test))
    print("train samples", len(samples_train))
    print("dev samples", len(samples_dev))

    for sample_index, sample in enumerate(tqdm(target_samples)):
        question = sample["question"]
        label = sample["label"]

        if label == "test":
            articles = "\n\n\n".join(get_full_texts(sample, sample_contexts_test, context_texts_test))
        elif label == "train":
            articles = "\n\n\n".join(get_full_texts(sample, sample_contexts_train, context_texts_train))
        else:
            articles = "\n\n\n".join(get_full_texts(sample, sample_contexts_dev, context_texts_dev))
        prompt_content_full = instruction_full
        prompt_content_full = prompt_content_full.replace("<question>", question)
        prompt_content_full = prompt_content_full.replace("<articles>", articles)
        sample["prompt_full_context"] = prompt_content_full

        snippets = "\n\n".join(get_proxy(sample))
        prompt_content_proxy = instruction_proxy
        prompt_content_proxy = prompt_content_proxy.replace("<question>", question)
        prompt_content_proxy = prompt_content_proxy.replace("<snippets>", snippets)
        sample["prompt_proxy_context"] = prompt_content_proxy

        target_samples[sample_index] = sample

    get_statistics(target_samples)
