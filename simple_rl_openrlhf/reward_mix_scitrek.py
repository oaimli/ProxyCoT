import torch
import re
import string
from collections import Counter
from typing import List

COMPILED_REGEX = re.compile(r"\\boxed\{([\s\S]*?)\}")
# numbers with commas inside
PATTERN_VALID_NUMBER = r'^(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?$'

def normalize_text(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    def round_number(text):
        try:
            tmp = float(text)
            return f"{tmp:.3f}"
        except ValueError:
            return text

    return round_number(white_space_fix(remove_articles(remove_punc(lower(s)))))


def exact_match(generation, reference):
    normalized_generation = normalize_text(generation)
    normalized_reference = normalize_text(reference)
    score = 0
    if normalized_generation == normalized_reference:
        score = 1
    return float(score)


def f1_score(generation, reference):
    """Token-level F1 without external deps (bag-of-words)."""
    g_tokens: List[str] = normalize_text(generation).split()
    r_tokens: List[str] = normalize_text(reference).split()
    if not g_tokens and not r_tokens:
        return 1.0
    if not g_tokens or not r_tokens:
        return 0.0
    # multiset overlap
    g_c = Counter(g_tokens)
    r_c = Counter(r_tokens)
    overlap = sum((g_c & r_c).values())
    if overlap == 0:
        return 0.0
    precision = overlap / max(1, sum(g_c.values()))
    recall = overlap / max(1, sum(r_c.values()))
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def reward_func(queries, prompts, labels, **kwargs):
    """
    Reward function for calculating rewards of model outputs.

    Args:
        queries (torch.Tensor): Complete text sequences containing prompts and responses
        prompts (torch.Tensor): Input prompt sequences
        labels (torch.Tensor): Ground truth answer sequences
        **kwargs: Additional optional parameters

    Returns:
        dict: A dictionary containing the following key-value pairs:
            - rewards: Reward values used for calculating advantage function
            - scores: Reward values in range [0,1] used for dynamic filtering
    """
    # extract answer from \boxed{} in query
    answers = []
    for query in queries:
        matches = COMPILED_REGEX.findall(query)
        answer = matches[-1].strip() if matches else ""
        print(f"the end of query: {query[-100:]}")
        answers.append(answer)

    reward = []
    for answer, label in zip(answers, labels):
        answer = answer.strip()
        answer = " ".join(answer.split())

        if bool(re.fullmatch(PATTERN_VALID_NUMBER, answer)):
            tmp = answer.replace(",", "")
            answer = f"{float(tmp):.3f}"
        else:
            items = []
            for tmp in answer.split(","):
                tmp = tmp.strip()
                try:
                    items.append(f"{float(tmp):.3f}")
                except ValueError:
                    items.append(tmp)
            answer = ", ".join(items)


        label = label.strip()
        label = " ".join(label.split())

        if bool(re.fullmatch(PATTERN_VALID_NUMBER, label)):
            tmp = label.replace(",", "")
            label = f"{float(tmp):.3f}"
        else:
            items = []
            for tmp in label.split(","):
                tmp = tmp.strip()
                try:
                    items.append(f"{float(tmp):.3f}")
                except ValueError:
                    items.append(tmp)
            label = ", ".join(items)

        em = exact_match(answer, label)
        f1 = f1_score(answer, label)

        mix = em + f1
        print(f"answer: {answer}, label: {label}, em: {em}, f1: {f1}")
        reward.append(mix)

    reward = torch.tensor(reward).float()
    return {
        "rewards": reward,
        "scores": reward,
    }