import string
import re

# numbers with commas inside
PATTERN_VALID_NUMBER = r'^(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?$'


def normalize_text(s):
    def white_space_fix(text):
        if len(text.split()) > 1:
            regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
            text = re.sub(regex, ' ', text)
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def round_number(text):
        # check if the text is a valid number with commas inside
        if bool(re.fullmatch(PATTERN_VALID_NUMBER, text)):
            text = text.replace(",", "")
            text = f"{float(text):.3f}"
            return text

        # check if the entire text or elements are numbers
        items = []
        for tmp in text.split(","):
            tmp = tmp.strip()
            try:
                tmp = f"{float(tmp):.3f}"
                items.append(tmp)
            except ValueError:
                items.append(tmp)
        text = ", ".join(items)
        return text

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(round_number(lower(s))))


def exact_match(generation, reference):
    normalized_generation = normalize_text(generation)
    normalized_reference = normalize_text(reference)
    em = 0.0
    if normalized_generation == normalized_reference:
        em = 1.0
    return em


if __name__ == "__main__":
    print(normalize_text("a b"))
    a = "1,2344"
    b = "12344"
    print(exact_match(a, b))