#!/usr/bin/env python3
"""
train_lora.py — LoRA fine-tune an open base model on the Masri dataset.

Designed to run on a single Colab/free-tier T4 (7B in 4-bit) or better.
Uses TRL's SFTTrainer + PEFT LoRA + bitsandbytes 4-bit quantization (QLoRA).

Usage:
  python3 train_lora.py \
      --base_model Qwen/Qwen3-8B \
      --train_file ../data/train.jsonl \
      --eval_file ../data/dev.jsonl \
      --output_dir ../out/masri-lora \
      --epochs 3

After training, merge + push with scripts/push_to_hub.py.
"""
import argparse
import json

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", default="Qwen/Qwen3-8B",
                    help="Base model repo id. Must have solid Arabic + broad Unicode "
                         "coverage (Coptic/Greek code points) in its tokenizer — see README. "
                         "Alternatives: tiiuae/Falcon3-Arabic-7B-Instruct (Arabic-native), "
                         "Qwen/Qwen3-4B (fits free Colab T4 more comfortably).")
    p.add_argument("--train_file", default="../data/train.jsonl")
    p.add_argument("--eval_file", default="../data/dev.jsonl")
    p.add_argument("--output_dir", default="../out/masri-lora")
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--max_seq_len", type=int, default=1024)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--no_4bit", action="store_true",
                    help="Disable 4-bit quantization (use if you have a big GPU / are doing full fine-tune of a small model).")
    p.add_argument("--resume_from_checkpoint", default=None,
                    help="Path to a checkpoint folder (e.g. ../out/masri-lora-v2/checkpoint-12) to resume an interrupted run.")
    p.add_argument("--push_to_hub", default=None,
                    help="Optional repo id, e.g. 'yourname/masri-qwen2.5-7b-lora', to push directly after training.")
    return p.parse_args()


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    if not args.no_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quant_config,
        device_map="auto",
        dtype=torch.float16,
    )
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
    )

    data_files = {"train": args.train_file}
    if args.eval_file:
        data_files["eval"] = args.eval_file
    raw_ds = load_dataset("json", data_files=data_files)

    def format_example(ex):
        text = tokenizer.apply_chat_template(
            ex["messages"], tokenize=False, add_generation_prompt=False, enable_thinking=False
        )
        return {"text": text}

    train_ds = raw_ds["train"].map(format_example, remove_columns=raw_ds["train"].column_names)
    eval_ds = None
    if "eval" in raw_ds:
        eval_ds = raw_ds["eval"].map(format_example, remove_columns=raw_ds["eval"].column_names)

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=5,
        eval_strategy="epoch" if eval_ds is not None else "no",
        save_strategy="steps",
        save_steps=10,
        save_total_limit=3,
        fp16=False,
        bf16=False,
        max_length=args.max_seq_len,
        dataset_text_field="text",
        packing=False,
        report_to="none",
        push_to_hub=bool(args.push_to_hub),
        hub_model_id=args.push_to_hub,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    with open(f"{args.output_dir}/train_args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    if args.push_to_hub:
        trainer.push_to_hub()
        print(f"Pushed adapter to https://huggingface.co/{args.push_to_hub}")

    print(f"Done. LoRA adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
