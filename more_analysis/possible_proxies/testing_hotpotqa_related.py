from vllm import LLM, SamplingParams
import jsonlines
from tqdm import tqdm
from transformers import AutoTokenizer
import json
import sys
import re
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import bm25s
import Stemmer

sys.path.append("../../")
from preparation.hotpotqa.data_loading import load_test


def get_metadata(sample):
    sentences = []
    target_titles = sample["supporting_facts"]["title"]
    for sentences_batch, title in zip(sample["context"]["sentences"], sample["context"]["title"]):
        if title in target_titles:
            sentences.extend(sentences_batch)
    return sentences


def get_context(sample, sample_contexts, context_sentences, supporting_sentences):
    sample_id = sample["id"]
    sample_question = sample["question"]
    sample_length = sample["length"]
    sentences_all = []
    for title in sample_contexts[f"{sample_id}_{sample_length}"]:
        sentences_all.extend(context_sentences[title])
    # if len(sentences_all) < 50:
    #     print(f"{sample_id}_{sample_length}")
    # if len(supporting_sentences) > 10:
    #     print(f"{sample_id}_{sample_length}")

    stemmer = Stemmer.Stemmer("english")
    corpus_tokens = bm25s.tokenize(sentences_all, stopwords="en", stemmer=stemmer)
    retriever = bm25s.BM25(corpus=sentences_all)
    retriever.index(corpus_tokens)
    query_tokens = bm25s.tokenize(sample_question, stopwords="en", stemmer=stemmer)
    if len(sentences_all) > 50:
        results_bm25, scores = retriever.retrieve(query_tokens, k=50)
        sentences_bm25 = results_bm25[0]
    else:
        sentences_bm25 = sentences_all
    # print(len(sentences_bm25), sentences_bm25[0])

    sentence_embeddings = []
    response = client.embeddings.create(
        input=sentences_bm25,
        model="text-embedding-3-small"
    )
    for result in response.data:
        sentence_embeddings.append(list(result.embedding))

    response = client.embeddings.create(
        input=sample_question,
        model="text-embedding-3-small"
    )
    question_embedding = list(response.data[0].embedding)

    sentence_similarities = list(cosine_similarity([question_embedding], sentence_embeddings)[0])
    assert len(sentence_similarities) == len(sentences_bm25)

    sentences_ranked = [tuple[1] for tuple in sorted(zip(sentence_similarities, sentences_bm25), reverse=True)]
    assert len(sentences_bm25) == len(sentences_ranked)
    print("QUESTION:", sample_question)
    print("SENTENCE:", [str(tmp) for tmp in sentences_ranked[:2]])
    sentences_related = [str(tmp) for tmp in sentences_ranked[:len(supporting_sentences)]]
    return sentences_related


if __name__ == "__main__":
    api_key = ""
    client = OpenAI(api_key=api_key)
    COMPILED_REGEX = re.compile(r"\\boxed\{([\s\S]*?)\}")
    dataset_dir = "../../preparation/hotpotqa/extended_hotpotqa"
    model_name_official = "checkpoint_hotpotqa_related/_actor/global_step140_hf"
    model_name_save = "qwen3_4b_instruct_0527_related"
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

    with open("context_sentences_hotpotqa_testing.json") as f:
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
                    related_sentences = get_context(working_sample, sample_contexts_test, context_sentences_testing, supporting_sentences)
                    snippets = "\n\n".join(related_sentences)
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