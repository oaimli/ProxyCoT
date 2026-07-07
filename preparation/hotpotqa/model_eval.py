from openai import OpenAI
import json


if __name__ == "__main__":
    model_name_official = "gpt-5-mini-2025-08-07"

    api_key = ""
    client = OpenAI(api_key=api_key)

    with open("eyeballing.json") as f:
        samples = json.load(f)
    for sample_index, sample in enumerate(samples):
        del sample["validity"]
        output_dict = client.chat.completions.create(
            model=model_name_official,
            messages=[
                {
                    "role": "user",
                    "content": f"{str(sample)}\n\n\nIs the given answer correct? If yes, output 1, otherwise output 0."}
                ],
            n=3
            )
        # print(output_dict)
        outputs = []
        for choice in output_dict.choices:
            outputs.append(choice.message.content)
        print(sample_index, outputs)