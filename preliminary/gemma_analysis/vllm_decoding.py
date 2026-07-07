from vllm import LLM, SamplingParams
import jsonlines
from tqdm import tqdm
from transformers import AutoTokenizer
import os
import sys
import re

sys.path.append("../../")
from preparation.scitrek.data_loading import load_test, load_articles, get_full_texts


if __name__ == "__main__":
    COMPILED_REGEX = re.compile(r"\\boxed\{([\s\S]*?)\}")
    dataset_dir = "../../../MDRL/bench"
    model_name_official = "google/gemma-3-4b-it"
    model_name_save = "gemma3_4b_it"
    target_mode = "full"
    vllm_tensor_parallel_size = 2
    vllm_max_model_length = 131072

    llm = LLM(model=model_name_official, max_model_len=vllm_max_model_length,
              tensor_parallel_size=vllm_tensor_parallel_size)
    #  max_tokens is for the maximum length for generation.
    sampling_params = SamplingParams(n=3, top_p=0.95, top_k=64, max_tokens=8192)

    articles_all = load_articles(articles_folder=dataset_dir + "/article/")

    # test data
    save_name = f"scitrek_{target_mode}_{model_name_save}_test.jsonl"
    if os.path.exists(save_name):
        samples_test = []
        with jsonlines.open(save_name) as reader:
            for line in reader:
                samples_test.append(line)
        print("existing test results loaded", len(samples_test))
    else:
        samples_test = []
        samples_test.extend(load_test(prefix="64k",
                                      samples_folder=dataset_dir + "/dataset/samples/final/",
                                      no_null=True, no_simple=True))
        samples_test.extend(load_test(prefix="128k",
                                      samples_folder=dataset_dir + "/dataset/samples/final/",
                                      no_null=True, no_simple=True))
        print("original test samples loaded", len(samples_test))

    tokenizer = AutoTokenizer.from_pretrained(model_name_official)
    for sample_index, sample in tqdm(enumerate(samples_test), total=len(samples_test),
                                     desc=f"scitrek_{model_name_save}_test"):
        if "generations" not in sample.keys():
            question = sample["question"]
            markdowns = get_full_texts(sample, articles_all)
            articles = "\n\n\n".join(markdowns)
            instruction = open("../../instructions/qwen_instruction_full_scitrek.txt").read()
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
            print(reasonings)
            sample["reasonings"] = reasonings
            generations = []
            for reasoning in reasonings:
                matches = COMPILED_REGEX.findall(reasoning)
                generation = matches[-1] if matches else ""
                generations.append(generation)
            sample["generations"] = generations
            print([sample["answer"]] + sample["generations"])
            samples_test[sample_index] = sample