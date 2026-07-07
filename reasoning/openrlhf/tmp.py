import jsonlines
from nltk.metrics import f_measure
from typing import List
import re
import string

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
            tmp = float(s)
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
    return score


def f1_score_one(generation, reference):
    score = f_measure(
        set(normalize_text(reference).split()), set(normalize_text(generation).split())
        )
    if score is None:  # the answer may be an empty string after normalizing
        score = 0.0
    return score

def f1_score_two(generation: str, reference: str) -> float:
    """Token-level F1 without external deps (bag-of-words)."""
    g_tokens: List[str] = normalize_text(generation).split()
    r_tokens: List[str] = normalize_text(reference).split()
    if not g_tokens and not r_tokens:
        return 1.0
    if not g_tokens or not r_tokens:
        return 0.0
    # multiset overlap
    from collections import Counter

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


if __name__ == "__main__":
    print(f1_score_one("A B", "B"))
    print(f1_score_two("A B", "B"))