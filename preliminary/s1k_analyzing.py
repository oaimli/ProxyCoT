from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
    dataset = load_dataset("simplescaling/s1K-1.1", split="train")
    question_lengths = []
    reasoning_lengths = []
    answer_lengths = []
    for sample in dataset:
        question = sample["question"]
        reasoning = sample["deepseek_thinking_trajectory"]
        answer = sample["solution"]
        tokenized_question = tokenizer.encode(question)
        question_lengths.append(len(tokenized_question))
        tokenized_solution = tokenizer.encode(reasoning)
        reasoning_lengths.append(len(tokenized_solution))
        tokenized_answer = tokenizer.encode(answer)
        answer_lengths.append(len(tokenized_answer))
    print("question lengths: ", np.mean(question_lengths), np.max(question_lengths))
    print("reasoning lengths: ", np.mean(reasoning_lengths), np.max(reasoning_lengths))
    print("answer lengths: ", np.mean(answer_lengths), np.max(answer_lengths))

