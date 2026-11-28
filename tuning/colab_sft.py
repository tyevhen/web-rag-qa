"""QLoRA of Qwen2.5-7B-Instruct on a CUDA GPU (Colab, Windows), the twin of tuning/lora.yaml (tuning/README.md, step 3).

    python colab_sft.py train --data DATA --out OUT   # resumes from the last OUT/checkpoint-* if any
    python colab_sft.py eval --data DATA --out OUT    # validation loss: OUT/adapter vs the base without it
    python colab_sft.py merge --out OUT [--fused DIR] # OUT/adapter + full-precision base -> DIR (/content/fused)

Same settings as lora.yaml (rank 16, scale 2 = alpha 32, dropout 0.05, constant lr 5e-5, effective
batch 4, 2 epochs, loss on the answer only), except every layer is trained: the 8 there was a 16 GB limit.
Runs on CUDA only (bitsandbytes): not part of the app, nothing imports it.
"""

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

BASE = "Qwen/Qwen2.5-7B-Instruct"


def to_prompt_completion(row: dict) -> dict:
    return {"prompt": row["messages"][:-1], "completion": row["messages"][-1:]}


def build(data: Path, out: Path, adapter: Path | None = None) -> SFTTrainer:
    """The trainer train and eval share: same data, 4-bit base and settings; a new LoRA unless `adapter` is given."""
    half = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16  # T4: no bf16 (is_bf16_supported() says yes: emulation)
    files = {"train": str(data / "train.jsonl"), "valid": str(data / "valid.jsonl")}
    ds = load_dataset("json", data_files=files).map(to_prompt_completion, remove_columns=["messages"])
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=half)
    model = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=quant, dtype=half)
    if adapter:
        model = PeftModel.from_pretrained(model, str(adapter))
    lora = None if adapter else LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear",
                                           task_type="CAUSAL_LM")
    args = SFTConfig(
        output_dir=str(out), max_length=4096, completion_only_loss=True,
        per_device_train_batch_size=1, per_device_eval_batch_size=1, gradient_accumulation_steps=4,
        num_train_epochs=2, learning_rate=5e-5, lr_scheduler_type="constant", optim="adamw_torch",
        bf16=half == torch.bfloat16, fp16=half == torch.float16, gradient_checkpointing=True,
        logging_steps=10, eval_strategy="steps", eval_steps=50, save_steps=50, save_total_limit=3,
        report_to="none", seed=0,
    )
    return SFTTrainer(model=model, args=args, train_dataset=ds["train"], eval_dataset=ds["valid"],
                      processing_class=AutoTokenizer.from_pretrained(BASE), peft_config=lora)


def train(data: Path, out: Path) -> None:
    trainer = build(data, out)
    for p in trainer.model.parameters():  # fp16's grad scaler needs fp32 trainables; LoRA came out bf16
        if p.requires_grad:
            p.data = p.data.float()
    trainer.train(resume_from_checkpoint=any(out.glob("checkpoint-*")))
    trainer.save_model(str(out / "adapter"))


def evaluate(data: Path, out: Path) -> None:
    """Validation loss of OUT/adapter, then of the base with it switched off: the curve's end points."""
    trainer = build(data, out, out / "adapter")
    tuned = trainer.evaluate()
    with trainer.model.disable_adapter():
        base = trainer.evaluate()
    for name, m in (("base", base), ("tuned", tuned)):
        print(f"{name}: eval_loss {m['eval_loss']:.4f}, token accuracy {m.get('eval_mean_token_accuracy', float('nan')):.4f}")


def merge(out: Path, fused_dir: Path) -> None:
    """Loads the whole bf16 model on the CPU: needs ~32 GB of RAM. Output ~15 GB (on Colab: the runtime's disk, not Drive)."""
    base = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cpu")
    fused = PeftModel.from_pretrained(base, str(out / "adapter")).merge_and_unload()
    fused.save_pretrained(str(fused_dir))
    AutoTokenizer.from_pretrained(BASE).save_pretrained(str(fused_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["train", "eval", "merge"])
    parser.add_argument("--data", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fused", type=Path, default=Path("/content/fused"))
    a = parser.parse_args()
    {"train": lambda: train(a.data, a.out), "eval": lambda: evaluate(a.data, a.out),
     "merge": lambda: merge(a.out, a.fused)}[a.step]()
