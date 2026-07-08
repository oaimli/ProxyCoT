from vllm import LLM, SamplingParams
import jsonlines
from tqdm import tqdm
from transformers import AutoTokenizer
import os
import sys
import re
import random

sys.path.append("../../")
from preparation.scitrek.data_loading import load_test, load_articles


def get_context(sample, articles_all):
    cluster_papers = []
    for paper_id in sample["articles"]:
        paper = articles_all[paper_id]
        paper["article_id"] = paper_id
        cluster_papers.append(paper)

    articles = []
    validity = True
    for paper in cluster_papers:
        article = []
        article_title = paper["title"]
        article.append(f"Article title:\n{article_title}")
        authors = paper["authors"]
        article.append(f"Authors:\n{authors}")
        markdown = paper["markdown"].lower()
        reference_index = markdown.rfind("reference")
        appendix_index = markdown.rfind("appendix")
        if appendix_index == -1:
            if reference_index != -1:
                reference_section = markdown[reference_index:]
            else:
                reference_section = ""
        else:
            if reference_index != -1:
                if appendix_index > reference_index:
                    reference_section = markdown[reference_index:appendix_index]
                else:
                    reference_section = markdown[reference_index:]
            else:
                reference_section = ""
        if reference_section == "":
            validity = False
            break
        else:
            article.append(f"Reference:\n{reference_section}")

        articles.append("\n\n".join(article))

    return articles, validity


if __name__ == "__main__":
    COMPILED_REGEX = re.compile(r"\\boxed\{([\s\S]*?)\}")
    dataset_dir = "../../../SciTrek/benchmark"
    model_name_official = "checkpoint_scitrek_fields/_actor/global_step60_hf"
    model_name_save = "qwen3_4b_instruct_0527_fields"
    target_mode = "proxy"
    inference_batch_size = 128
    vllm_tensor_parallel_size = 4
    vllm_max_model_length = 65536 + 2048
    instruction = open("../../instructions/instruction_proxy_scitrek.txt").read()
    llm = LLM(model=model_name_official, max_model_len=vllm_max_model_length,
              tensor_parallel_size=vllm_tensor_parallel_size)
    #  max_tokens is for the maximum length for generation.
    sampling_params = SamplingParams(n=3, temperature=0.7, top_p=0.8, top_k=20, min_p=0, max_tokens=2048)
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
    # random.seed(42)
    # samples_test = random.sample(samples_test, 200)

    working_instances = []
    for sample_index, sample in tqdm(enumerate(samples_test), total=len(samples_test),
                                     desc=f"scitrek_{model_name_save}_test"):
        fields_articles, validity = get_context(sample, articles_all)
        if "generations" not in sample.keys() and validity:
            working_instances.append({"index": sample_index, "sample": sample, "fields": fields_articles})
            if len(working_instances) == inference_batch_size or (len(working_instances) > 0 and sample_index == len(samples_test) - 1):
                inputs = []
                for working_instance in working_instances:
                    working_sample = working_instance["sample"]
                    question = working_sample["question"]
                    fields_articles = working_instance["fields"]
                    prompt_content = instruction
                    prompt_content = prompt_content.replace("<question>", question)
                    prompt_content = prompt_content.replace("<articles>", "\n\n\n".join(fields_articles))
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
