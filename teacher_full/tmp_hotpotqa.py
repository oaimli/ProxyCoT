import jsonlines


if __name__ == '__main__':
    samples_train = []
    with jsonlines.open('hotpotqa_full_qwen3_235b_a22b_thinking_0527_train_0.jsonl') as reader:
        for line in reader:
            samples_train.append(line)

    processed_samples = []
    valid_samples = []
    for sample_index, sample in enumerate(samples_train):
        if "generations" in sample.keys():
            processed_samples.append(sample_index)
            valid_generations = []
            valid_reasonings = []
            generations = sample['generations']
            reasonings = sample['reasonings']
            for reasoning, generation in zip(reasonings, generations):
                if generation != "":
                    valid_generations.append(generation)
                    valid_reasonings.append(reasoning)
            if len(valid_generations) > 0 and len(valid_reasonings) > 0:
                valid_samples.append(sample_index)
    print("valid samples:", len(valid_samples))
    print("processed samples:", len(processed_samples))


    samples_train = []
    with jsonlines.open('hotpotqa_full_qwen3_235b_a22b_thinking_0527_train_1.jsonl') as reader:
        for line in reader:
            samples_train.append(line)

    processed_samples = []
    valid_samples = []
    for sample_index, sample in enumerate(samples_train):
        if "generations" in sample.keys():
            processed_samples.append(sample_index)
            valid_generations = []
            valid_reasonings = []
            generations = sample['generations']
            reasonings = sample['reasonings']
            for reasoning, generation in zip(reasonings, generations):
                if generation != "":
                    valid_generations.append(generation)
                    valid_reasonings.append(reasoning)
            if len(valid_generations) > 0 and len(valid_reasonings) > 0:
                valid_samples.append(sample_index)
    print("valid samples:", len(valid_samples))
    print("processed samples:", len(processed_samples))


    samples_dev = []
    with jsonlines.open('hotpotqa_full_qwen3_235b_a22b_thinking_0527_dev_1.jsonl') as reader:
        for line in reader:
            samples_dev.append(line)

    processed_samples = []
    valid_samples = []
    for sample_index, sample in enumerate(samples_dev):
        if "generations" in sample.keys():
            processed_samples.append(sample_index)
            valid_generations = []
            valid_reasonings = []
            generations = sample['generations']
            reasonings = sample['reasonings']
            for reasoning, generation in zip(reasonings, generations):
                if generation != "":
                    valid_generations.append(generation)
                    valid_reasonings.append(reasoning)
            if len(valid_generations) > 0 and len(valid_reasonings) > 0:
                valid_samples.append(sample_index)
    print("valid samples:", len(valid_samples))
    print("processed samples:", len(processed_samples))


    samples_dev = []
    with jsonlines.open('hotpotqa_full_qwen3_235b_a22b_thinking_0527_dev.jsonl') as reader:
        for line in reader:
            samples_dev.append(line)

    processed_samples = []
    valid_samples = []
    for sample_index, sample in enumerate(samples_dev):
        if "generations" in sample.keys():
            processed_samples.append(sample_index)
            valid_generations = []
            valid_reasonings = []
            generations = sample['generations']
            reasonings = sample['reasonings']
            for reasoning, generation in zip(reasonings, generations):
                if generation != "":
                    valid_generations.append(generation)
                    valid_reasonings.append(reasoning)
            if len(valid_generations) > 0 and len(valid_reasonings) > 0:
                valid_samples.append(sample_index)
    print("valid samples:", len(valid_samples))
    print("processed samples:", len(processed_samples))