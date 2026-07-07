import re
from vllm import LLM, SamplingParams
import jsonlines
from transformers import AutoTokenizer
import random


if __name__ == "__main__":
    COMPILED_REGEX = re.compile(r"\\boxed\{([\s\S]*?)\}")
    model_name_official = "Qwen/Qwen3-4B-Instruct-2507"
    model_name_save = "qwen3_4b_instruct_0527"
    vllm_tensor_parallel_size = 4
    vllm_max_model_length = 133120

    llm = LLM(model=model_name_official, max_model_len=vllm_max_model_length,
              tensor_parallel_size=vllm_tensor_parallel_size)
    tokenizer = AutoTokenizer.from_pretrained(model_name_official)
    #  max_tokens is for the maximum length for generation.
    sampling_params = SamplingParams(n=3, temperature=0.6, top_p=0.95, top_k=20, min_p=0, max_tokens=2048)

    # ['example_id', 'context_titles']
    samples_contexts = {}
    with jsonlines.open("data/test_124k_example_contexts.jsonl") as reader:
        for line in reader:
            samples_contexts[line["example_id"]] = line["context_titles"]

    # ['title', 'text']
    context_texts = {}
    with jsonlines.open("data/test_124k_context_texts.jsonl") as reader:
        for line in reader:
            context_texts[line["title"]] = line["text"]

    # ['id', 'question', 'answer', 'type', 'level', 'supporting_facts', 'context']
    samples = []
    with jsonlines.open("data/test_124k.jsonl") as reader:
        for line in reader:
            samples.append(line)
    print("samples", len(samples))

    save_name = f"hotpotqa_{model_name_save}_test.jsonl"
    for sample_index, sample in enumerate(samples):
        context_titles = samples_contexts[sample["id"]]
        random.shuffle(context_titles)
        context_articles = []
        for title in context_titles:
            context_articles.append(context_texts[title])

        question = sample["question"]
        answer = sample["answer"]
        type = sample["type"]
        level = sample["level"]
        assert type == "bridge" and level == "hard"

        articles = "\n\n\n".join(context_articles)
        instruction = open("../../instructions/qwen_instruction_full_hotpotqa.txt").read()
        instruction = instruction.replace("<question>", question)
        instruction = instruction.replace("<articles>", articles)
        prompt_content = instruction
        conversation = [{"role": "user", "content": prompt_content}]
        text = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
            )
        conversation_outputs = llm.generate([text], sampling_params, use_tqdm=False)
        # print(conversation_outputs)
        reasonings = []
        for conversation_output in conversation_outputs:
            for tmp in conversation_output.outputs:
                reasonings.append(tmp.text)
        # print(reasonings)
        sample["reasonings_long"] = reasonings
        generations = []
        for reasoning in reasonings:
            matches = COMPILED_REGEX.findall(reasoning)
            generation = matches[-1] if matches else ""
            generations.append(generation)
        sample["generations_long"] = generations
        print([answer] + sample["generations_long"])


        target_titles = sample["supporting_facts"]["title"]
        sentences = []
        for sentences_batch, title in zip(sample["context"]["sentences"], sample["context"]["title"]):
            if title in target_titles:
                sentences.extend(sentences_batch)
        assert len(sentences) > 0
        snippets = "\n".join(sentences)
        instruction = open("../../instructions/qwen_instruction_proxy_hotpotqa.txt").read()
        instruction = instruction.replace("<question>", question)
        instruction = instruction.replace("<snippets>", snippets)
        prompt_content = instruction
        conversation = [{"role": "user", "content": prompt_content}]
        text = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
            )
        conversation_outputs = llm.generate([text], sampling_params, use_tqdm=False)
        # print(conversation_outputs)
        reasonings = []
        for conversation_output in conversation_outputs:
            for tmp in conversation_output.outputs:
                reasonings.append(tmp.text)
        # print(reasonings)
        sample["reasonings_proxy"] = reasonings
        generations = []
        for reasoning in reasonings:
            matches = COMPILED_REGEX.findall(reasoning)
            generation = matches[-1] if matches else ""
            generations.append(generation)
        sample["generations_proxy"] = generations
        print([answer] + sample["generations_proxy"])


        instruction = open("../../instructions/qwen_instruction_none_hotpotqa.txt").read()
        instruction = instruction.replace("<question>", question)
        prompt_content = instruction
        conversation = [{"role": "user", "content": prompt_content}]
        text = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
            )
        conversation_outputs = llm.generate([text], sampling_params, use_tqdm=False)
        # print(conversation_outputs)
        reasonings = []
        for conversation_output in conversation_outputs:
            for tmp in conversation_output.outputs:
                reasonings.append(tmp.text)
        # print(reasonings)
        sample["reasonings_none"] = reasonings
        generations = []
        for reasoning in reasonings:
            matches = COMPILED_REGEX.findall(reasoning)
            generation = matches[-1] if matches else ""
            generations.append(generation)
        sample["generations_none"] = generations
        print([answer] + sample["generations_none"])

        samples[sample_index] = sample
        if sample_index % 10 == 0 and sample_index != 0:
            with jsonlines.open(save_name, "w") as writer:
                writer.write_all(samples)
    with jsonlines.open(save_name, "w") as writer:
        writer.write_all(samples)


