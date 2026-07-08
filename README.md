# ProxyCoT: Long-Context Reasoning Through Proxy-Based Chain-of-Thought Tuning
[![ACL](https://img.shields.io/badge/ACL2026-Main-darkred)](https://aclanthology.org/2026.acl-long.1917/) [![arxiv](https://img.shields.io/badge/arXiv-2605.20201-lightgrey)](https://arxiv.org/pdf/2509.21028) [![code](https://img.shields.io/badge/GitHub-ProxyCoT-darkblue)](https://github.com/oaimli/ProxyCoT) 

### What is in this repository
```
/
├── case_study/             --> (Manually check data samples from SciTrek and HotpotQA)
├── eyeballing/             --> (Eyeballing model generated reasoning traces and answers)
├── figure_plots/           --> (Plot the figures in the paper)
├── grounding/              --> (Use the teacher-generated reasoning traces to run SFT training, which is the second stage of our training framework)
├── instructions/           --> (All prompt templates that we used in experiments for both full and proxy contexts)
├── more_analysis/          --> (Analysis on reasoning traces and robustness of proxy contexts)
├── ood/                    --> (Out-of-domain evaluation of our models on Loong)   
├── preliminary/            --> (Experimental trials on open-source models)
├── preparation/            --> (Preparation code for data)
├── reasoning/opnrlhf/      --> (RLVR on short contexts to get the teacher model in the first stage of ProxyCoT-RL)
├── simple_distillation/    --> (SFT distillation on full contexts with full-context-based reasoning traces from a teacher model)
├── simple_rl_openrlhf/     --> (RLVR on full contexts)
├── simple_sft/             --> (SFT on full contexts, with only generating the answers)
├── teacher_full/           --> (Running teacher inference on full contexts to get reasoning traces) 
├── teacher_proxy/          --> (Running teacher inference on proxy contexts to get reasoning traces)
├── teacher_table/          --> (Running teacher inference on database-table contexts to get reasoning traces for SciTrek)
├── testing/                --> (Testing zero-short prompting with popular long-context models)
├── utils/                  --> (Evaluation code for SciTrek based on EM and F1, and HotpotQA based on Model-as-Judge)
└── README.md               --> (This readme file)
```

### Dataset preparation

To run the experiments for ProxyCoT, you need to first download the data for SciTrek and HotpotQA (extended).

- For SciTrek, clone the repo and download the data following the instructions in https://github.com/oaimli/SciTrek.
- For HotpotQA, download the data from [Google Drive](https://drive.google.com/drive/folders/13Ia7sQapLQbBoBWizAMLrVdxsC2TxtIQ?usp=sharing), and replace ProxyCoT/preparation/hotpotqa/extended_hotpotqa with the downloaded folder of extended_hotpotqa. Please refer to our paper for more details on how we extended HotpotQA.

The two datasets can also be downloaded from Hugging Face for further use. This is only for further use of the datasets, not reproduction of our experiments.

```python
train_samples = load_dataset("oaimli/proxycot-scitrek", split="train")
dev_samples = load_dataset("oaimli/proxycot-scitrek", split="val")
test_samples = load_dataset("oaimli/proxycot-scitrek", split="test")

for sample in train_samples:
    question = sample["question"]
    answer = sample["answer"]
    metadata = sample["metadata"]
    articles = sample["articles"]
    instruction_proxy = sample["instruction_proxy"]
    instruction_full = sample["instruction_full"]

    instruction_proxy = instruction_proxy.replace("<question>", question)
    instruction_proxy = instruction_proxy.replace("<articles>", "\n\n\n".join(metadata))
    conversation_proxy = [{"role": "user", "content": instruction_proxy}]

    instruction_full = instruction_full.replace("<question>", question)
    instruction_full = instruction_full.replace("<articles>", "\n\n\n".join(articles))
    conversation_full = [{"role": "user", "content": instruction_full}]
```

```python
train_samples = load_dataset("oaimli/proxycot-hotpotqa", split="train")
dev_samples = load_dataset("oaimli/proxycot-hotpotqa", split="val")
test_samples = load_dataset("oaimli/proxycot-hotpotqa", split="test")

for sample in train_samples:
    question = sample["question"]
    answer = sample["answer"]
    metadata = sample["metadata"]
    articles = sample["articles"]
    instruction_proxy = sample["instruction_proxy"]
    instruction_full = sample["instruction_full"]

    instruction_proxy = instruction_proxy.replace("<question>", question)
    instruction_proxy = instruction_proxy.replace("<articles>", "\n\n\n".join(metadata))
    conversation_proxy = [{"role": "user", "content": instruction_proxy}]

    instruction_full = instruction_full.replace("<question>", question)
    instruction_full = instruction_full.replace("<articles>", "\n\n\n".join(articles))
    conversation_full = [{"role": "user", "content": instruction_full}]
```
### Experimental environment preparation

Our experiments require the following libraries:

```bash
transformers==4.57.3
vllm==0.13.0
jsonlines==4.0.0
torch==2.9.0
datasets==4.5.0
wandb==0.24.0
liger-kernel==0.6.4
flash_attn==2.8.3
ring-flash-attn==0.1.8
bitsandbytes==0.48.2
accelerate==1.12.0
openrlhf==0.9.1
deepspeed==0.18.3
ray[default]==2.48.0
```
If you have issues in running OpenRLHF-0.9.1, please use our customised OpenRLHF for the project, [Qwen3](https://github.com/oaimli/OpenRLHF/tree/0.9.1-qwen3-all), [Gemma3-GRPO](https://github.com/oaimli/OpenRLHF/tree/0.9.1-gemma3-grpo), and [Gemma3-SFT](https://github.com/oaimli/OpenRLHF/tree/0.9.1-gemma3-sft). 
