import random
import numpy as np
from vllm import LLM, SamplingParams
from tqdm import tqdm
from transformers import AutoTokenizer
import sys
import re
import jsonlines

sys.path.append("../")
from preparation.scitrek.data_loading import load_articles, get_full_texts, load_test
from utils.answer_eval import exact_match, f1_score, pre_processing


def evaluating(generation_batches, references):
    generation_batches_processed, references_processed = pre_processing(generation_batches, references)
    scores = {}
    scores.update(exact_match(generation_batches_processed, references_processed))
    scores.update(f1_score(generation_batches_processed, references_processed))
    return scores


if __name__ == "__main__":
    COMPILED_REGEX = re.compile(r"\\boxed\{([\s\S]*?)\}")
    dataset_dir = "../../SciTrek/benchmark"
    model_name_official = "agurung/SFT-gemma-3-4b-it"
    model_name_save = "gemma3_4b_it_reinforcement_alex"
    target_mode = "full"
    inference_batch_size = 1024
    vllm_tensor_parallel_size = 4
    vllm_max_model_length = 131072
    save_name = f"scitrek_{target_mode}_{model_name_save}_test.jsonl"
    instruction = open("../instructions/instruction_full_scitrek.txt").read()

    llm = LLM(model=model_name_official, max_model_len=vllm_max_model_length,
              tensor_parallel_size=vllm_tensor_parallel_size)
    #  max_tokens is for the maximum length for generation.
    sampling_params = SamplingParams(temperature=1.0, top_p=0.95, top_k=64, max_tokens=2048)

    articles_all = load_articles(articles_folder=dataset_dir + "/article/")

    # test data
    samples_test = []
    samples_test.extend(load_test(prefix="64k",
                                  samples_folder=dataset_dir + "/dataset/samples/final/",
                                  no_null=True, no_simple=True))
    samples_test.extend(load_test(prefix="128k",
                                  samples_folder=dataset_dir + "/dataset/samples/final/",
                                  no_null=True, no_simple=True))
    print("original test samples loaded", len(samples_test))
    # samples_test = random.sample(samples_test, 100)

    tokenizer = AutoTokenizer.from_pretrained(model_name_official)

    working_instances = []
    for sample_index, sample in tqdm(enumerate(samples_test), total=len(samples_test),
                                     desc=f"scitrek_test"):
        if "generations" not in sample.keys():
            working_instances.append({"index": sample_index, "sample": sample})
            if len(working_instances) == inference_batch_size or (len(working_instances) > 0 and sample_index == len(samples_test) - 1):
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
                    samples_test[working_sample_index] = sample_tmp

                with jsonlines.open(save_name, "w") as writer:
                    writer.write_all(samples_test)

                working_instances = []

    # evaluation
    generation_batches = []
    answers = []
    for sample in samples_test:
        generation_batches.append(sample["generations"])
        answers.append(sample["answer"])
    scores = evaluating(generation_batches, answers)
    metrics = []
    values_all = []
    for metric, values_tmp in scores.items():
        metrics.append(metric)
        values_all.append(str(round(np.mean(values_tmp), 3)))
    print(metrics, "/".join(values_all))
