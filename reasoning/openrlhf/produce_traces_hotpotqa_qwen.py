from vllm import LLM, SamplingParams
import jsonlines
from tqdm import tqdm
from transformers import AutoTokenizer
import os
import sys
import re

sys.path.append("../../")
from preparation.hotpotqa.data_loading import load_train, load_dev, get_proxy


if __name__ == "__main__":
    COMPILED_REGEX = re.compile(r"\\boxed\{([\s\S]*?)\}")
    dataset_dir = "../../preparation/hotpotqa/data"
    model_name_official = "checkpoint_hotpotqa_qwen/_actor/global_step150_hf"
    model_name_save = "qwen3_4b_instruct_0527_rl_proxy"
    target_mode = "proxy"
    inference_batch_size = 128
    vllm_tensor_parallel_size = 4
    vllm_max_model_length = 4096 + 2048
    instruction = open("../../instructions/qwen_instruction_proxy_hotpotqa.txt").read()

    llm = LLM(model=model_name_official, max_model_len=vllm_max_model_length,
              tensor_parallel_size=vllm_tensor_parallel_size)
    #  max_tokens is for the maximum length for generation.
    sampling_params = SamplingParams(n=3, temperature=0.7, top_p=0.8, top_k=20, min_p=0, max_tokens=2048)
    tokenizer = AutoTokenizer.from_pretrained(model_name_official)

    # train data
    samples_train, _, _ = load_train(data_folder=dataset_dir)
    save_name = f"hotpotqa_{target_mode}_{model_name_save}_train.jsonl"
    if os.path.exists(save_name):
        samples_train = []
        with jsonlines.open(save_name) as reader:
            for line in reader:
                samples_train.append(line)
        print("existing train results loaded", len(samples_train))
    else:
        print("original train samples loaded", len(samples_train))

    working_instances = []
    for sample_index, sample in tqdm(enumerate(samples_train), total=len(samples_train),
                                     desc=f"hotpotqa_{model_name_save}_train"):
        if "generations" not in sample.keys():
            working_instances.append({"index": sample_index, "sample": sample})
            if len(working_instances) == inference_batch_size or (len(working_instances) > 0 and sample_index == len(samples_train) - 1):
                inputs = []
                for working_instance in working_instances:
                    working_sample = working_instance["sample"]
                    question = working_sample["question"]
                    snippets = "\n\n".join(get_proxy(working_sample))
                    prompt_content = instruction
                    prompt_content = prompt_content.replace("<question>", question)
                    prompt_content = prompt_content.replace("<snippets>", snippets)
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
                    print(reasonings)
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


    # dev data
    samples_dev, _, _ = load_dev(data_folder=dataset_dir)
    save_name = f"hotpotqa_{target_mode}_{model_name_save}_dev.jsonl"
    if os.path.exists(save_name):
        samples_dev = []
        with jsonlines.open(save_name) as reader:
            for line in reader:
                samples_dev.append(line)
        print("existing dev results loaded", len(samples_dev))
    else:
        print("original dev samples loaded", len(samples_dev))

    working_instances = []
    for sample_index, sample in tqdm(enumerate(samples_dev), total=len(samples_dev),
                                     desc=f"hotpotqa_{model_name_save}_dev"):
        if "generations" not in sample.keys():
            working_instances.append({"index": sample_index, "sample": sample})
            if len(working_instances) == inference_batch_size or sample_index == len(samples_dev) - 1:
                inputs = []
                for working_instance in working_instances:
                    working_sample = working_instance["sample"]
                    question = working_sample["question"]
                    snippets = "\n\n".join(get_proxy(working_sample))
                    prompt_content = instruction
                    prompt_content = prompt_content.replace("<question>", question)
                    prompt_content = prompt_content.replace("<snippets>", snippets)
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
                    print(reasonings)
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
                    samples_dev[working_sample_index] = sample_tmp

                with jsonlines.open(save_name, "w") as writer:
                    writer.write_all(samples_dev)
                working_instances = []
