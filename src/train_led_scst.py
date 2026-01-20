# src/train_led_scst.py
import os, gc
from pathlib import Path
import numpy as np
import torch
from torch.optim import AdamW
from tqdm import tqdm

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from rouge_score import rouge_scorer


# =========================
# PATHS
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_JSONL = PROJECT_ROOT / "data" / "processed" / "dataset" / "pdf_train.jsonl"
VAL_JSONL   = PROJECT_ROOT / "data" / "processed" / "dataset" / "pdf_val.jsonl"

# Start SCST from your already SFT-trained adapter
SFT_ADAPTER_PATH = PROJECT_ROOT / "led_large_ckpt" / "final_adapter"

OUT_DIR = PROJECT_ROOT / "led_large_ckpt_scst"
OUT_ADAPTER = OUT_DIR / "final_adapter"

# =========================
# MODEL + SCST SETTINGS
# =========================
MODEL_ID = "allenai/led-large-16384-arxiv"

MAX_INPUT_LEN = 2048
MAX_NEW_TOKENS = 256
MAX_CHARS = 12000

# Sampling params for exploration
TOP_P = 0.90
TEMPERATURE = 1.0

LR = 3e-5
EPOCHS = 1

GRAD_ACC = 8
MAX_TRAIN_EXAMPLES = 400   # start small; increase later (e.g., 1000+)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def clear_mem():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_inputs(tokenizer, text: str):
    text = text[:MAX_CHARS]
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LEN,
        padding=False
    )
    return inputs


@torch.no_grad()
def generate_summary(model, tokenizer, inputs, do_sample: bool):
    # LED global attention: first token global
    input_ids = inputs["input_ids"].to(DEVICE)
    attention_mask = inputs["attention_mask"].to(DEVICE)
    global_attention_mask = torch.zeros_like(input_ids)
    global_attention_mask[:, 0] = 1

    gen_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        global_attention_mask=global_attention_mask,
        max_new_tokens=MAX_NEW_TOKENS,
        no_repeat_ngram_size=4,
        repetition_penalty=1.10
    )

    if do_sample:
        gen_kwargs.update(dict(
            do_sample=True,
            top_p=TOP_P,
            temperature=TEMPERATURE,
            num_beams=1
        ))
    else:
        gen_kwargs.update(dict(
            do_sample=False,
            num_beams=4
        ))

    out = model.generate(**gen_kwargs)
    return out


def rougeL_f1(scorer, pred: str, ref: str) -> float:
    return scorer.score(ref, pred)["rougeL"].fmeasure


def main():
    assert TRAIN_JSONL.exists(), f"Missing: {TRAIN_JSONL}"
    assert SFT_ADAPTER_PATH.exists(), f"Missing: {SFT_ADAPTER_PATH}"

    print("📂 Loading dataset...")
    ds = load_dataset("json", data_files={"train": str(TRAIN_JSONL), "validation": str(VAL_JSONL)})
    train_ds = ds["train"]

    if MAX_TRAIN_EXAMPLES:
        train_ds = train_ds.select(range(min(MAX_TRAIN_EXAMPLES, len(train_ds))))

    print(train_ds)

    print("🔤 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print("🧠 Loading base model + SFT adapter (fp16)...")
    base = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to(DEVICE)

    # IMPORTANT: make adapter trainable for SCST
    model = PeftModel.from_pretrained(base, str(SFT_ADAPTER_PATH), is_trainable=True).to(DEVICE)

    # sanity check
    model.print_trainable_parameters()
    trainable = [p for p in model.parameters() if p.requires_grad]
    print("Trainable tensors:", len(trainable))
    assert len(trainable) > 0, "No trainable params. Adapter is frozen."


    # SCST training stability
    model.train()
    model.config.use_cache = False

    optimizer = AdamW(model.parameters(), lr=LR)

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))

    print("🚀 Starting SCST reward training...")
    step = 0
    running_adv = []
    running_rl = []

    for epoch in range(EPOCHS):
        pbar = tqdm(train_ds, desc=f"SCST Epoch {epoch+1}/{EPOCHS}")
        optimizer.zero_grad(set_to_none=True)

        for row in pbar:
            article = row["article"]
            ref_abs = row["abstract"]

            # Build inputs
            inputs = build_inputs(tokenizer, article)

            # 1) Greedy
            greedy_ids = generate_summary(model, tokenizer, inputs, do_sample=False)
            greedy_txt = tokenizer.decode(greedy_ids[0], skip_special_tokens=True)

            # 2) Sampled
            sampled_ids = generate_summary(model, tokenizer, inputs, do_sample=True)
            sampled_txt = tokenizer.decode(sampled_ids[0], skip_special_tokens=True)

            # Rewards
            r_g = rougeL_f1(scorer, greedy_txt, ref_abs)
            r_s = rougeL_f1(scorer, sampled_txt, ref_abs)

            adv = (r_s - r_g)

            # 3) RL loss = adv * NLL(sampled)
            # Convert sampled to labels for teacher forcing
            labels = sampled_ids.clone()
            labels[labels == tokenizer.pad_token_id] = -100

            # forward pass to compute NLL on sampled sequence
            with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
                input_ids = inputs["input_ids"].to(DEVICE)
                attention_mask = inputs["attention_mask"].to(DEVICE)
                global_attention_mask = torch.zeros_like(input_ids)
                global_attention_mask[:, 0] = 1

                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    global_attention_mask=global_attention_mask,
                    labels=labels.to(DEVICE),
                )
                nll = out.loss  # mean NLL over tokens

                # SCST objective: minimize nll if adv > 0, maximize if adv < 0
                rl_loss = (adv * nll)

            scaler.scale(rl_loss / GRAD_ACC).backward()

            step += 1
            running_adv.append(adv)
            running_rl.append(float(rl_loss.detach().cpu()))

            if step % GRAD_ACC == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            if len(running_adv) >= 20:
                pbar.set_postfix({
                    "adv(avg)": f"{np.mean(running_adv[-20:]):.4f}",
                    "rl_loss(avg)": f"{np.mean(running_rl[-20:]):.4f}",
                    "r_g": f"{r_g:.3f}",
                    "r_s": f"{r_s:.3f}",
                })

    OUT_ADAPTER.mkdir(parents=True, exist_ok=True)
    print("💾 Saving SCST adapter to:", OUT_ADAPTER)
    model.save_pretrained(str(OUT_ADAPTER))
    tokenizer.save_pretrained(str(OUT_ADAPTER))

    print("✅ SCST DONE")


if __name__ == "__main__":
    clear_mem()
    main()
