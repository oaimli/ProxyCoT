from vllm import LLM, SamplingParams
import jsonlines
from tqdm import tqdm
from transformers import AutoTokenizer
import os
import sys
import re

sys.path.append("../")
from preparation.scitrek.data_loading import load_train, load_articles, get_full_texts



if __name__ == "__main__":
    COMPILED_REGEX = re.compile(r"\\boxed\{([\s\S]*?)\}")
    dataset_dir = "../../SciTrek/benchmark"
    model_name_official = "Qwen/Qwen3-235B-A22B-Thinking-2507"
    model_name_save = "qwen3_235b_a22b_thinking_0527"
    target_mode = "full"
    inference_batch_size = 128
    vllm_tensor_parallel_size = 4
    vllm_max_model_length = 131072 + 10240
    instruction = open("../instructions/instruction_full_scitrek.txt").read()
    llm = LLM(model=model_name_official, max_model_len=vllm_max_model_length,
              tensor_parallel_size=vllm_tensor_parallel_size)
    #  max_tokens is for the maximum length for generation.
    sampling_params = SamplingParams(n=1, temperature=0.6, top_p=0.95, top_k=20, min_p=0, max_tokens=10240)
    tokenizer = AutoTokenizer.from_pretrained(model_name_official)
    articles_all = load_articles(articles_folder=dataset_dir + "/article/")

    # training data
    save_name = f"scitrek_{target_mode}_{model_name_save}_train_3.jsonl"
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

    working_instances = []
    for sample_index, sample in tqdm(enumerate(samples_train), total=len(samples_train),
                                     desc=f"scitrek_{model_name_save}_train"):
        if "generations" not in sample.keys() and sample_index > 6600:
            working_instances.append({"index": sample_index, "sample": sample})
            if len(working_instances) == inference_batch_size or (len(working_instances) > 0 and sample_index == len(samples_train) - 1):
                inputs = []
                for working_instance in working_instances:
                    working_sample = working_instance["sample"]
                    question = working_sample["question"]
                    markdowns = get_full_texts(working_sample, articles_all)
                    articles = "\n\n\n".join(markdowns)
                    prompt_content = instruction
                    prompt_content = prompt_content.replace("<question>", question)
                    prompt_content = prompt_content.replace("<articles>", articles)
                    conversation = [{"role": "user", "content": prompt_content}]
                    text = tokenizer.apply_chat_template(
                        conversation,
                        tokenize=False,
                        add_generation_prompt=True
                        )
                    inputs.append(text)
                conversation_outputs = llm.generate(inputs, sampling_params, use_tqdm=True)
                working_outputs = []
                for conversation_output, working_instance in zip(conversation_outputs, working_instances):
                    # print(conversation_outputs)
                    reasonings = []
                    for tmp in conversation_output.outputs:
                        reasonings.append(tmp.text)
                    # print(reasonings)
                    generations = []
                    for reasoning in reasonings:
                        matches = COMPILED_REGEX.findall(reasoning)
                        generation = matches[-1] if matches else ""
                        generations.append(generation)
                    print([working_instance["sample"]["answer"]] + generations)
                    working_outputs.append([reasonings, generations])

                for working_output, working_instance in zip(working_outputs, working_instances):
                    working_sample_index = working_instance["index"]
                    sample_tmp = working_instance["sample"]
                    sample_tmp["reasonings"] = working_output[0]
                    sample_tmp["generations"] = working_output[1]
                    samples_train[working_sample_index] = sample_tmp

                with jsonlines.open(save_name, "w") as writer:
                    writer.write_all(samples_train)
                working_instances = []