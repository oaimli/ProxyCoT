import random

import jsonlines
import json

samples_dev = []
with jsonlines.open("scitrek_proxy_qwen3_235b_a22b_thinking_0527_dev.jsonl") as reader:
    for line in reader:
        samples_dev.append(line)

samples_eyeball = random.sample(samples_dev, 20)

with open("eyeballing.json", "w") as f:
    json.dump(samples_eyeball, f, indent=4)