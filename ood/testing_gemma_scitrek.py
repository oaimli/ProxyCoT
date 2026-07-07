from vllm import LLM, SamplingParams
import jsonlines
from tqdm import tqdm
from transformers import AutoTokenizer
import re
import os


if __name__ == "__main__":
    COMPILED_REGEX = re.compile(r"\\boxed\{([\s\S]*?)\}")
    article_folder = "loong/doc"
    model_name_official = "../grounding/saved_tmp/checkpoint_scitrek_reinforcement_gemma_5/global_step475_hf"
    model_name_save = "gemma3_4b_it_scitrek_475"
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
    sampling_params = SamplingParams(n=3, temperature=1.0, top_p=0.95, top_k=64, max_tokens=2048)

    # test data
    save_name = f"loong_{model_name_save}_test.jsonl"
    working_instances = []
    for sample_index, sample in tqdm(enumerate(samples_test), total=len(samples_test),
                                     desc=f"loong_{model_name_save}_test_alex"):
        articles = []
        for doc in sample["doc"]:
            if sample["type"] == "paper":
                with open(f"{article_folder}/{sample['type']}/{doc}") as f:
                    articles.append(f.read())
            elif sample["type"] == "financial":
                target_files = []
                for file in os.listdir(f"{article_folder}/{sample['type']}"):
                    if doc in file:
                        target_files.append(file)
                with open(f"{article_folder}/{sample['type']}/{target_files[0]}") as f:
                    articles.append(f.read())
            else:
                print("error in sample type.", sample["type"])
        question = sample["question"]
        instruction = sample["instruction"]
        # "{docs}\n\n{instruction}\n\n{question}"
        prompt_template = sample["prompt_template"]
        prompt_template = prompt_template.replace("{docs}", "\n\n".join(articles))
        prompt_template = prompt_template.replace("{instruction}", instruction)
        prompt_template = prompt_template.replace("{question}", question)
        prompt_content = prompt_template + "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
        messages = [{"role": "user", "content": prompt_content}]
        text_tokenized = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True)
        print(len(text_tokenized))
        if len(text_tokenized) < 131072 - 2048:
            working_instances.append({"index": sample_index, "sample": sample, "conversation": messages})

        if len(working_instances) == inference_batch_size or (len(working_instances) > 0 and sample_index == len(samples_test) - 1):
            inputs = []
            for working_instance in working_instances:
                text = tokenizer.apply_chat_template(
                    working_instance["conversation"],
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
