from vllm import LLM, SamplingParams
import jsonlines
from tqdm import tqdm
from transformers import AutoTokenizer
import re
import os


if __name__ == "__main__":
    COMPILED_REGEX = re.compile(r"\\boxed\{([\s\S]*?)\}")
    article_folder = "loong/doc"
    model_name_official = "oaimli/longtune_scitrek_grounding_reinforcement_qwen_5_500"
    model_name_save = "qwen3_4b_instruct_0527_scitrek_500"
    inference_batch_size = 128
    vllm_tensor_parallel_size = 4
    vllm_max_model_length = 131072

    samples_test = []
    with jsonlines.open("valid_loong.jsonl") as reader:
        for line in reader:
            samples_test.append(line)
    print("samples_test", len(samples_test))

    llm = LLM(model=model_name_official, max_model_len=vllm_max_model_length,
              tensor_parallel_size=vllm_tensor_parallel_size)
    #  max_tokens is for the maximum length for generation.
    tokenizer = AutoTokenizer.from_pretrained(model_name_official)
    sampling_params = SamplingParams(n=3, temperature=0.7, top_p=0.8, top_k=20, min_p=0, max_tokens=32768)

    # test data
    save_name = f"loong_{model_name_save}_test.jsonl"
    working_instances = []
    for sample_index, sample in tqdm(enumerate(samples_test), total=len(samples_test),
                                     desc=f"loong_{model_name_save}_test"):
        working_instances.append({"index": sample_index, "sample": sample})
        if len(working_instances) == inference_batch_size or (len(working_instances) > 0 and sample_index == len(samples_test) - 1):
            inputs = []
            for working_instance in working_instances:
                working_sample = working_instance["sample"]
                articles = []
                for doc in working_sample["doc"]:
                    if working_sample["type"] == "paper":
                        with open(f"{article_folder}/{working_sample['type']}/{doc}") as f:
                            articles.append(f.read())
                    elif working_sample["type"] == "financial":
                        target_files = []
                        for file in os.listdir(f"{article_folder}/{working_sample['type']}"):
                            if doc in file:
                                target_files.append(file)
                        with open(f"{article_folder}/{working_sample['type']}/{target_files[0]}") as f:
                            articles.append(f.read())
                    else:
                        print("error in sample type.", working_sample["type"])
                question = working_sample["question"]
                instruction = working_sample["instruction"]
                # "{docs}\n\n{instruction}\n\n{question}"
                prompt_template = working_sample["prompt_template"]
                prompt_template = prompt_template.replace("{docs}", "\n\n".join(articles))
                prompt_template = prompt_template.replace("{instruction}", instruction)
                prompt_template = prompt_template.replace("{question}", question)
                prompt_content = prompt_template + "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
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
