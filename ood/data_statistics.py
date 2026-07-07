import jsonlines


if __name__ == '__main__':
    samples_all = []
    samples_paper = []
    samples_finance = []
    with jsonlines.open('valid_loong.jsonl') as reader:
        for line in reader:
            samples_all.append(line)
            if line["type"] == "paper":
                samples_paper.append(line)
            if line["type"] == "financial":
                samples_finance.append(line)
    print("samples all", len(samples_all), "samples paper", len(samples_paper), "samples finance", len(samples_finance))