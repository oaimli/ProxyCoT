from vllm import LLM, SamplingParams
import jsonlines
from tqdm import tqdm
from transformers import AutoTokenizer
import os
import sys
import re

sys.path.append("../")
from preparation.scitrek.data_loading import load_test, load_articles, get_full_texts


if __name__ == "__main__":
    COMPILED_REGEX = re.compile(r"\\boxed\{([\s\S]*?)\}")
    dataset_dir = "../../SciTrek/benchmark"
    model_name_official = "Qwen/Qwen3-32B"
    model_name_save = "qwen3_32b"
    target_mode = "full"
    inference_batch_size = 64
    vllm_tensor_parallel_size = 4
    vllm_max_model_length = 131072
    instruction = open("../instructions/instruction_full_scitrek.txt").read()
    llm = LLM(model=model_name_official, max_model_len=vllm_max_model_length,
              tensor_parallel_size=vllm_tensor_parallel_size, rope_scaling={"type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768})
    #  max_tokens is for the maximum length for generation.
    sampling_params = SamplingParams(n=3, temperature=0.6, top_p=0.95, top_k=20, min_p=0, max_tokens=32768)
    tokenizer = AutoTokenizer.from_pretrained(model_name_official)
    articles_all = load_articles(articles_folder=dataset_dir + "/article/")

    # test data
    save_name = f"scitrek_{target_mode}_{model_name_save}_test.jsonl"
    samples_test = []
    samples_test.extend(load_test(prefix="64k",
                                  samples_folder=dataset_dir + "/dataset/samples/final/",
                                  no_null=True, no_simple=True))
    samples_test.extend(load_test(prefix="128k",
                                  samples_folder=dataset_dir + "/dataset/samples/final/",
                                  no_null=True, no_simple=True))
    print("original test samples loaded", len(samples_test))

    working_instances = []
    for sample_index, sample in tqdm(enumerate(samples_test), total=len(samples_test),
                                     desc=f"scitrek_{model_name_save}_test"):
        if "generations" not in sample.keys():
            working_instances.append({"index": sample_index, "sample": sample})
            if len(working_instances) == inference_batch_size or (
                    len(working_instances) > 0 and sample_index == len(samples_test) - 1):
                inputs = []
                for working_instance in working_instances:
                    working_sample = working_instance["sample"]
                    question = working_sample["question"]
                    articles = "\n\n\n".join(get_full_texts(working_sample, articles_all))
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
                    samples_test[working_sample_index] = sample_tmp

                with jsonlines.open(save_name, "w") as writer:
                    writer.write_all(samples_test)

                working_instances = []