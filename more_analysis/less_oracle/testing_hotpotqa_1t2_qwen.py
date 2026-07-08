from vllm import LLM, SamplingParams
import jsonlines
from tqdm import tqdm
from transformers import AutoTokenizer
import random
import sys
import re
import json

sys.path.append("../../")
from preparation.hotpotqa.data_loading import load_test


def get_metadata(sample):
    sentences = []
    target_titles = sample["supporting_facts"]["title"]
    for sentences_batch, title in zip(sample["context"]["sentences"], sample["context"]["title"]):
        if title in target_titles:
            sentences.extend(sentences_batch)
    return sentences


def get_context(sample, sample_contexts, supporting_sentences, context_sentences):
    sample_id = sample["id"]
    sample_length = sample["length"]
    sentences_all = []
    for title in sample_contexts[f"{sample_id}_{sample_length}"]:
        sentences_all.extend(context_sentences[title])
    noise_sentences = random.sample(sentences_all, len(supporting_sentences) * 2)
    mixed_sentences = supporting_sentences + noise_sentences
    random.shuffle(mixed_sentences)
    return mixed_sentences


if __name__ == "__main__":
    COMPILED_REGEX = re.compile(r"\\boxed\{([\s\S]*?)\}")
    dataset_dir = "../../preparation/hotpotqa/extended_hotpotqa"
    model_name_official = ""
    model_name_save = "qwen3_4b_instruct_0527_1t2"
    target_mode = "proxy"
    inference_batch_size = 128
    vllm_tensor_parallel_size = 4
    vllm_max_model_length = 65536
    instruction = open("../../instructions/instruction_proxy_hotpotqa.txt").read()
    llm = LLM(model=model_name_official, max_model_len=vllm_max_model_length,
              tensor_parallel_size=vllm_tensor_parallel_size)
    #  max_tokens is for the maximum length for generation.
    sampling_params = SamplingParams(n=3, temperature=0.7, top_p=0.8, top_k=20, min_p=0, max_tokens=32768)
    tokenizer = AutoTokenizer.from_pretrained(model_name_official)
    samples_test, sample_contexts_test, _ = load_test(data_folder=dataset_dir)

    with open("../possible_proxies/context_sentences_hotpotqa_testing.json") as f:
        context_sentences_testing = json.load(f)

    # test data
    save_name = f"hotpotqa_{target_mode}_{model_name_save}_test.jsonl"

    working_instances = []
    for sample_index, sample in tqdm(enumerate(samples_test), total=len(samples_test),
                                     desc=f"hotpotqa_{model_name_save}_test"):
        if "generations" not in sample.keys():
            working_instances.append({"index": sample_index, "sample": sample})
            if len(working_instances) == inference_batch_size or (len(working_instances) > 0 and sample_index == len(samples_test) - 1):
                inputs = []
                for working_instance in working_instances:
                    working_sample = working_instance["sample"]
                    question = working_sample["question"]
                    supporting_sentences = get_metadata(working_sample)
                    mixed_sentences = get_context(working_sample, sample_contexts_test, supporting_sentences, context_sentences_testing)
                    snippets = "\n\n".join(mixed_sentences)
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
                    samples_test[working_sample_index] = sample_tmp

                with jsonlines.open(save_name, "w") as writer:
                    writer.write_all(samples_test)
                working_instances = []