import os
import jsonlines
import numpy as np
from transformers import AutoTokenizer
import re


def analyzing_length(samples):
    reasoning_lengths = []
    for sample in samples:
        reasonings = sample["reasonings"]
        generations = sample["generations"]
        answer = sample["answer"]
        for reasoning, generation in zip(reasonings, generations):
            generation = str(generation).strip()
            generation = " ".join(generation.split())
            generation = ", ".join([tmp.strip() for tmp in generation.split(",")])
            if generation == answer:
                tokenized_reasoning = tokenizer.encode(reasoning)
                reasoning_lengths.append(len(tokenized_reasoning))
    print("reasoning length:", len(reasoning_lengths), np.min(reasoning_lengths), np.mean(reasoning_lengths), np.max(reasoning_lengths))


def analyzing_answer_positions(samples):
    generations = []
    for sample in samples:
        reasonings = sample["reasonings"]
        for reasoning in reasonings:
            matches = COMPILED_REGEX.findall(reasoning)
            if matches:
                generations.append(matches)
            else:
                generations.append([])
    early_answers = []
    for generation in generations:
        if len(generation) > len(set(generation)):
            early_answers.append(generation)
    print("early answers", len(early_answers))


if __name__ == "__main__":
    COMPILED_REGEX = re.compile(r"\\boxed\{([\s\S]*?)\}")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
    result_folder = "../testing"
    output_files = []
    for file in os.listdir(result_folder):
        if file.endswith(".jsonl"):
            output_files.append(file)
    for generation_file in output_files:
        print(generation_file)
        samples = []
        with jsonlines.open(os.path.join(result_folder, generation_file)) as reader:
            for line in reader:
                samples.append(line)
        # analyzing_length(samples)
        analyzing_answer_positions(samples)