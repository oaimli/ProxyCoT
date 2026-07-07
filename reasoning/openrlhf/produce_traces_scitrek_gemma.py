from vllm import LLM, SamplingParams
import jsonlines
from tqdm import tqdm
from transformers import AutoTokenizer
import os
import sys
import re

sys.path.append("../../")
from preparation.scitrek.data_loading import load_train, load_articles, get_proxy



if __name__ == "__main__":
    COMPILED_REGEX = re.compile(r"\\boxed\{([\s\S]*?)\}")
    dataset_dir = "../../../MDRL/bench"
    model_name_official = "checkpoint_scitrek_gemma/_actor/global_step250_hf"
    model_name_save = "gemma3_4b_it_rl_proxy"
    target_mode = "proxy"
    vllm_tensor_parallel_size = 4
    vllm_max_model_length = 4096 + 2048

    llm = LLM(model=model_name_official, max_model_len=vllm_max_model_length,
              tensor_parallel_size=vllm_tensor_parallel_size)
    #  max_tokens is for the maximum length for generation.
    sampling_params = SamplingParams(n=3, top_p=0.95, top_k=64, max_tokens=2048)

    articles_all = load_articles(articles_folder=dataset_dir + "/article/")

    # training data
    save_name = f"scitrek_{target_mode}_{model_name_save}_train.jsonl"
    if os.path.exists(save_name):
        samples_train = []
        with jsonlines.open(save_name) as reader:
            for line in reader:
                samples_train.append(line)
        print("existing training results loaded", len(samples_train))
    else:
        samples_train, _ = load_train(prefixes=["64k", "128k"], samples_folder=dataset_dir + "/dataset/samples/final/",
                                 no_null=True, no_simple=True)
        print("original training samples loaded", len(samples_train))

    tokenizer = AutoTokenizer.from_pretrained(model_name_official)
    for sample_index, sample in tqdm(enumerate(samples_train), total=len(samples_train),
                                     desc=f"scitrek_{model_name_save}_train"):
        if "generations" not in sample.keys():
            question = sample["question"]
            articles = get_proxy(sample, articles_all)
            instruction = open("../../testing/instruction_proxy_scitrek.txt").read()
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
            sample["reasonings"] = reasonings
            generations = []
            for reasoning in reasonings:
                matches = COMPILED_REGEX.findall(reasoning)
                generation = matches[-1].strip() if matches else ""
                generations.append(generation)
            sample["generations"] = generations
            print([sample["answer"]] + sample["generations"])
            samples_train[sample_index] = sample
            if sample_index % 10 == 0:
                with jsonlines.open(save_name, "w") as writer:
                    writer.write_all(samples_train)

    with jsonlines.open(save_name, "w") as writer:
        writer.write_all(samples_train)

    # dev data
    save_name = f"scitrek_{target_mode}_{model_name_save}_dev.jsonl"
    if os.path.exists(save_name):
        samples_dev = []
        with jsonlines.open(save_name) as reader:
            for line in reader:
                samples_dev.append(line)
        print("existing dev results loaded", len(samples_dev))
    else:
        _, samples_dev = load_train(prefixes=["64k", "128k"],
                                      samples_folder=dataset_dir + "/dataset/samples/final/",
                                      no_null=True, no_simple=True)
        print("original dev samples loaded", len(samples_dev))

    tokenizer = AutoTokenizer.from_pretrained(model_name_official)
    for sample_index, sample in tqdm(enumerate(samples_dev), total=len(samples_dev),
                                     desc=f"scitrek_{model_name_save}_dev"):
        if "generations" not in sample.keys():
            question = sample["question"]
            articles = get_proxy(sample, articles_all)
            instruction = open("../../testing/instruction_proxy_scitrek.txt").read()
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
            sample["reasonings"] = reasonings
            generations = []
            for reasoning in reasonings:
                matches = COMPILED_REGEX.findall(reasoning)
                generation = matches[-1].strip() if matches else ""
                generations.append(generation)
            sample["generations"] = generations
            print([sample["answer"]] + sample["generations"])
            samples_dev[sample_index] = sample
            if sample_index % 10 == 0:
                with jsonlines.open(save_name, "w") as writer:
                    writer.write_all(samples_dev)

    with jsonlines.open(save_name, "w") as writer:
        writer.write_all(samples_dev)
