# src/train_led.py
import gc
import os
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType


# =========================
# PATHS (project-root safe)
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_JSONL = PROJECT_ROOT / "data" / "processed" / "dataset" / "pdf_train.jsonl"
VAL_JSONL   = PROJECT_ROOT / "data" / "processed" / "dataset" / "pdf_val.jsonl"

OUTPUT_DIR  = PROJECT_ROOT / "led_large_ckpt"

# =========================
# MODEL CONFIG (T4-safe)
# =========================
MODEL_ID = "allenai/led-large-16384-arxiv"

# T4-safe defaults for LED-large:
MAX_INPUT_LEN  = 3072
MAX_TARGET_LEN = 350

# =========================
# TRAINING CONFIG
# =========================
EPOCHS = 2
LR = 5e-5

TRAIN_BS = 1
EVAL_BS  = 1
GRAD_ACC = 16

LOG_STEPS  = 50
SAVE_STEPS = 500  # save periodically; adjust as you like

# IMPORTANT:
# Running evaluation with generation during training can OOM on LED-large.
# We'll train WITHOUT eval and evaluate after training in a separate script/cell.
DO_EVAL_DURING_TRAINING = False

# LoRA config (lighter for LED-large)
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1


def clear_mem():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    assert TRAIN_JSONL.exists(), f"Missing: {TRAIN_JSONL}"
    assert VAL_JSONL.exists(), f"Missing: {VAL_JSONL}"

    print("📂 Loading dataset...")
    ds = load_dataset(
        "json",
        data_files={"train": str(TRAIN_JSONL), "validation": str(VAL_JSONL)},
    )
    print(ds)

    print("🔤 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    def preprocess(batch):
        inputs = tokenizer(
            batch["article"],
            truncation=True,
            padding="max_length",
            max_length=MAX_INPUT_LEN,
        )
        outputs = tokenizer(
            batch["abstract"],
            truncation=True,
            padding="max_length",
            max_length=MAX_TARGET_LEN,
        )

        labels = []
        for seq in outputs["input_ids"]:
            labels.append([t if t != tokenizer.pad_token_id else -100 for t in seq])

        # global attention: first token attends to all
        global_attention = [[1] + [0] * (MAX_INPUT_LEN - 1) for _ in inputs["input_ids"]]

        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "global_attention_mask": global_attention,
            "labels": labels,
        }

    print("⚙️ Tokenizing...")
    tokenized = ds.map(
        preprocess,
        batched=True,
        batch_size=2,
        remove_columns=ds["train"].column_names,
    )

    print("🧠 Loading LED-large model (fp16) + applying LoRA...")
    clear_mem()

    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
    )

    # Apply LoRA
    peft_cfg = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "v_proj"],
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16)

    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    model = get_peft_model(model, peft_cfg)
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    # model = get_peft_model(model, peft_cfg)
    # model.print_trainable_parameters()

    # # Memory saver: gradient checkpointing
    # model.gradient_checkpointing_enable()
    # model.config.use_cache = False  # required w/ checkpointing in training

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # -------------------------
    # TrainingArguments (compat: evaluation_strategy vs eval_strategy)
    # -------------------------
    common_kwargs = dict(
        output_dir=str(OUTPUT_DIR),
        overwrite_output_dir=True,

        per_device_train_batch_size=TRAIN_BS,
        per_device_eval_batch_size=EVAL_BS,
        gradient_accumulation_steps=GRAD_ACC,

        learning_rate=LR,
        num_train_epochs=EPOCHS,

        logging_steps=LOG_STEPS,

        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=2,

        fp16=True,
        remove_unused_columns=False,
        report_to="none",

        dataloader_num_workers=0,
    )

    # Disable eval during training by default (T4-safe)
    if DO_EVAL_DURING_TRAINING:
        # If you enable this, expect higher VRAM use due to generation.
        # Keep it rare if you turn it on.
        common_kwargs.update(dict(
            predict_with_generate=True,
            generation_max_length=MAX_TARGET_LEN,
            generation_num_beams=4,
        ))
        try:
            args = Seq2SeqTrainingArguments(evaluation_strategy="steps", eval_steps=1000, **common_kwargs)
        except TypeError:
            args = Seq2SeqTrainingArguments(eval_strategy="steps", eval_steps=1000, **common_kwargs)
        eval_dataset = tokenized["validation"]
    else:
        # No evaluation during training
        common_kwargs.update(dict(
            predict_with_generate=False,
        ))
        try:
            args = Seq2SeqTrainingArguments(evaluation_strategy="no", **common_kwargs)
        except TypeError:
            args = Seq2SeqTrainingArguments(eval_strategy="no", **common_kwargs)
        eval_dataset = None

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("🚀 Starting training...")
    trainer.train()

    save_path = OUTPUT_DIR / "final_adapter"
    save_path.mkdir(parents=True, exist_ok=True)

    print("💾 Saving adapter + tokenizer to:", save_path)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    print("✅ DONE")


if __name__ == "__main__":
    main()
