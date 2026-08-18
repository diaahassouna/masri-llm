#!/usr/bin/env python3
"""
augment_with_claude.py — OPTIONAL but recommended step.

119 hand-grounded examples (see dataset_stats.json) is not enough for a base
model to reliably internalize a novel, exception-heavy orthography like Masri
Tier 2 — especially the memorized closed lists (word-final gemination,
tha-merger direction, irregular loanword plurals). This script uses the
Claude API, fed your exact rule set as grounding, to generate many more
(Arabic sentence -> Masri Tier 2) pairs, which you should spot-check before
merging into train.jsonl.

Requires: pip install anthropic
          export ANTHROPIC_API_KEY=sk-ant-...

Usage:
  python3 augment_with_claude.py --n 400 --out ../data/train_augmented.jsonl

Then manually review a sample (at minimum the tha_merger, gemination, and
loanword_irregular_plural categories, since those are memorized-not-derived
and easiest for a generator to get wrong), and concatenate the good rows
onto train.jsonl before fine-tuning.
"""
import argparse
import json
import os
import random
from pathlib import Path

import anthropic

SRC = Path(__file__).parent.parent / "source"
DATA = Path(__file__).parent.parent / "data"

SYSTEM_PROMPT = (SRC / "masri_tier2_system_prompt.md").read_text(encoding="utf-8")

GEN_INSTRUCTIONS = """You are helping build a fine-tuning dataset for the Masri Tier 2 writing \
system described in the system prompt above, created by Diaa Hassouna (CC BY 4.0).

Generate {batch_size} NEW, natural Urban Cairene Egyptian Arabic sentences (in Arabic script), \
each paired with its correct Masri Tier 2 transliteration, following every rule in the system \
prompt exactly. Vary topic, length (mix short phrases and longer sentences), and which rules \
each example exercises — deliberately include some sentences that combine multiple rules \
(gemination + ayin + definite article in one sentence, etc.), matching the difficulty style of \
the "stacks the hardest combination" test items you'd expect in an eval set.

Do NOT reuse any of these already-covered example words/sentences verbatim: {avoid_list}

Return ONLY a JSON array, no prose, no markdown fences, in this exact shape:
[{{"arabic": "...", "masri": "...", "category": "gemination|definite_article|glottal_stop|ayin|q_hamza_merger|p_b_v_f|short_vowels|loanword|tha_merger|homograph|conversational"}}, ...]
"""


def load_avoid_list():
    avoid = []
    if (DATA / "train.jsonl").exists():
        for line in (DATA / "train.jsonl").read_text(encoding="utf-8").splitlines():
            ex = json.loads(line)
            for m in ex["messages"]:
                if m["role"] == "assistant":
                    avoid.append(m["content"][:40])
    return avoid[:60]  # keep prompt short


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="total examples to generate")
    ap.add_argument("--batch_size", type=int, default=25)
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--out", default=str(DATA / "train_augmented.jsonl"))
    args = ap.parse_args()

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    avoid_list = load_avoid_list()

    all_rows = []
    n_batches = (args.n + args.batch_size - 1) // args.batch_size
    for i in range(n_batches):
        msg = client.messages.create(
            model=args.model,
            max_tokens=4000,
            system=SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": GEN_INSTRUCTIONS.format(
                    batch_size=args.batch_size, avoid_list=avoid_list
                ),
            }],
        )
        raw = "".join(b.text for b in msg.content if b.type == "text").strip()
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
        try:
            batch = json.loads(raw)
        except json.JSONDecodeError:
            print(f"[batch {i}] failed to parse, skipping")
            continue
        all_rows.extend(batch)
        print(f"[batch {i+1}/{n_batches}] +{len(batch)} examples (total {len(all_rows)})")

    with open(args.out, "w", encoding="utf-8") as f:
        for row in all_rows:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"حوّل الجملة دي للمصرية (Tier 2): {row['arabic']}"},
                {"role": "assistant", "content": row["masri"]},
            ]
            f.write(json.dumps({"messages": messages, "category": row.get("category", "augmented")},
                                ensure_ascii=False) + "\n")

    print(f"\nWrote {len(all_rows)} augmented examples to {args.out}")
    print("REVIEW THESE BEFORE TRAINING — especially tha_merger, gemination, and loanword rows,")
    print("since those rules are memorized closed lists, not derivable patterns, and a generator")
    print("model can confidently produce a plausible-looking but wrong spelling.")


if __name__ == "__main__":
    main()
