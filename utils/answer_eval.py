import string
import re
import numpy as np
from collections import Counter
from typing import List

# numbers with commas inside
PATTERN_VALID_NUMBER = r'^(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?$'


def pre_processing_single(generation_batch, reference):
    generation_batch_processed = []
    for generation in generation_batch:
        generation = str(generation) # the generated answer may be a list
        generation = generation.strip()
        generation = " ".join(generation.split())

        if bool(re.fullmatch(PATTERN_VALID_NUMBER, generation)):
            tmp = generation.replace(",", "")
            generation = f"{float(tmp):.3f}"
        else:
            items = []
            for tmp in generation.split(","):
                tmp = tmp.strip()
                try:
                    items.append(f"{float(tmp):.3f}")
                except ValueError:
                    items.append(tmp)
            generation = ", ".join(items)
        generation_batch_processed.append(generation)

    reference = reference.strip()
    reference = " ".join(reference.split())
    if bool(re.fullmatch(PATTERN_VALID_NUMBER, reference)):
        tmp = reference.replace(",", "")
        reference = f"{float(tmp):.3f}"
    else:
        items = []
        for tmp in reference.split(","):
            tmp = tmp.strip()
            try:
                items.append(f"{float(tmp):.3f}")
            except ValueError:
                items.append(tmp)
        reference = ", ".join(items)

    return generation_batch_processed, reference


def pre_processing(generation_batches, references):
    generation_batches_processed = []
    for generation_batch in generation_batches:
        generation_batch_processed = []
        for generation in generation_batch:
            generation = str(generation) # the generated answer may be a list
            generation = generation.strip()
            generation = " ".join(generation.split())

            if bool(re.fullmatch(PATTERN_VALID_NUMBER, generation)):
                tmp = generation.replace(",", "")
                generation = f"{float(tmp):.3f}"
            else:
                items = []
                for tmp in generation.split(","):
                    tmp = tmp.strip()
                    try:
                        items.append(f"{float(tmp):.3f}")
                    except ValueError:
                        items.append(tmp)
                generation = ", ".join(items)
                # generation = ", ".join([tmp.strip() for tmp in generation.split(",")])
            generation_batch_processed.append(generation)
        generation_batches_processed.append(generation_batch_processed)

    references_processed = []
    for reference in references:
        reference = reference.strip()
        reference = " ".join(reference.split())
        if bool(re.fullmatch(PATTERN_VALID_NUMBER, reference)):
            tmp = reference.replace(",", "")
            reference = f"{float(tmp):.3f}"
        else:
            items = []
            for tmp in reference.split(","):
                tmp = tmp.strip()
                try:
                    items.append(f"{float(tmp):.3f}")
                except ValueError:
                    items.append(tmp)
            reference = ", ".join(items)
        references_processed.append(reference)

    return generation_batches_processed, references_processed


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


def exact_match_single(generation, reference):
    normalized_generation = normalize_text(generation)
    normalized_reference = normalize_text(reference)
    em = 0.0
    if normalized_generation == normalized_reference:
        em = 1.0
    return em


def exact_match(generation_batches, references):
    ems = []
    for generation_batch, reference in zip(generation_batches, references):
        ems_batch = []
        for generation in generation_batch:
            ems_batch.append(exact_match_single(generation, reference))
            # print(normalized_generation, normalized_reference, em)
        # print(generation_batch)
        # print(reference)
        # print(ems_batch)
        ems.append(np.average(ems_batch))
    return {'em': ems}


def f1_score_single(generation, reference):
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


def f1_score(generation_batches, references):
    f1_scores = []
    for generation_batch, reference in zip(generation_batches, references):
        f1_scores_batch = []
        for generation in generation_batch:
            score = f1_score_single(generation, reference)
            f1_scores_batch.append(score)
        # print(generation_batch)
        # print(reference)
        # print(f1_scores_batch)
        f1_scores.append(np.average(f1_scores_batch))
    return {"f1": f1_scores}


if __name__ == "__main__":
    a = [["2,1,234", "1234", "1,234"], ["2,345", "2345", "2,345"]]
    b = ["1234", "2345"]
    print(pre_processing(a, b))