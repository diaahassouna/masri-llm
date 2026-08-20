#!/usr/bin/env python3
"""Run this in your training environment (has network access to HF) before retraining
to confirm how many examples were truncated at the old max_seq_len=1024, and to pick
a safe new value."""
import json
from transformers import AutoTokenizer

BASE_MODEL = "Qwen/Qwen3-8B"
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

for split in ["train", "dev"]:
    lines = open(f"masri-llm/data/{split}.jsonl", encoding="utf-8").readlines()
    lengths = []
    for line in lines:
        ex = json.loads(line)
        text = tokenizer.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=False)
        n = len(tokenizer(text)["input_ids"])
        lengths.append(n)
    over_1024 = sum(1 for n in lengths if n > 1024)
    print(f"{split}: n={len(lengths)}  min={min(lengths)}  max={max(lengths)}  "
          f"over_1024={over_1024} ({100*over_1024/len(lengths):.0f}%)")
