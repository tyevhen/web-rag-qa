# QLoRA tuning of the generator (Qwen2.5-7B-Instruct)

Trains a LoRA adapter on the answer step (`tuning/export.py` writes the examples), merges it and serves
it from Ollama as `web-rag-llm-ft`. Everything large lives on the external SSD:

```bash
ROOT="/Volumes/ADATA SD700/web-rag-tuning"
```

1. The 4-bit base to train on (~4.3 GB; training on a frozen 4-bit base is what makes it QLoRA) and the
   full-precision one (~15 GB) to merge into:
   ```bash
   HF_HUB_CACHE="$ROOT/hf-cache" uv run hf download mlx-community/Qwen2.5-7B-Instruct-4bit --local-dir "$ROOT/base-4bit"
   HF_HUB_CACHE="$ROOT/hf-cache" uv run hf download Qwen/Qwen2.5-7B-Instruct --local-dir "$ROOT/base"
   ```
2. Split, grouped by source passage:
   ```bash
   uv run python -m tuning.split --tokenizer "$ROOT/base-4bit" --out "$ROOT/data"
   ```
3. Train with `tuning.train`, not `mlx_lm.lora` directly. Plain `mlx_lm.lora` locks 12.7 GB of 16 GB for
   the GPU, and on 2026-11-27 that hung macOS until the watchdog restarted the laptop. The wrapper takes
   the same arguments, locks at most 6 GB, computes the loss in 512-token pieces, and exits (code 3) when
   macOS has under 15% memory free. A step on a 3k-token row uses ~7 GB (MLX peak) and left 16% free.

   Before a run, quit everything except one terminal: Ollama (`brew services stop ollama`), Docker, the
   dashboard, VS Code and browsers. `sysctl kern.memorystatus_level` should say 80 or more. Keep the laptop
   on power with the lid open, and don't use it heavily while it trains.

   Smoke run (about 15 minutes; it should print 20 loss lines and save `adapters-smoke/adapters.safetensors`):
   ```bash
   caffeinate -i uv run python -m tuning.train -c tuning/lora.yaml --model "$ROOT/base-4bit" --data "$ROOT/data" --adapter-path "$ROOT/adapters-smoke" --iters 20 --steps-per-eval 1000
   ```
   Full run (settings in `tuning/lora.yaml`; ~14 hours on an M1 at ~13 ms per token; an adapter is saved every 50 steps):
   ```bash
   caffeinate -i uv run python -m tuning.train -c tuning/lora.yaml --model "$ROOT/base-4bit" --data "$ROOT/data" --adapter-path "$ROOT/adapters" 2>&1 | tee "$ROOT/train.log"
   ```
   If it stops on the memory guard, resume from the last checkpoint with
   `--resume-adapter-file "$ROOT/adapters/NNNNNNN_adapters.safetensors" --iters <remaining>`.
   To use an earlier checkpoint (validation loss rose later), copy `NNNNNNN_adapters.safetensors` over `adapters.safetensors`.
   **Or on a CUDA GPU** (Colab, or a Windows PC with an NVIDIA card): `tuning/colab_sft.py` runs the same
   settings with TRL and PEFT on every layer, takes rows up to 4096 tokens (split with `--max-tokens 4096`)
   and saves a PEFT adapter, not an MLX one. A 7B 4-bit base needs ~8.5 GB of GPU memory. On a Colab T4
   (fp16) it takes ~4 hours. `tuning/data-4k` is that split, committed so a training machine needs only a
   clone. On Windows (PowerShell, from the repo root):
   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"   # uv, once
   $T = "$HOME\web-rag-tuning"
   uv venv .venv-sft --python 3.13
   uv pip install --python .venv-sft torch --index-url https://download.pytorch.org/whl/cu128
   uv pip install --python .venv-sft "trl==1.19.0" "peft==0.22.0" "bitsandbytes==0.50.2" "transformers==5.22.1" datasets
   .venv-sft\Scripts\python -c "import torch; print(torch.cuda.get_device_name(), torch.cuda.get_device_capability())"
   .venv-sft\Scripts\python tuning\colab_sft.py train --data tuning\data-4k --out $T\run
   ```
   The check should print the card and a capability like `(12, 0)`; 8 or higher trains in bf16. The first run
   downloads the 15 GB base into `~\.cache\huggingface`. Keep the PC plugged in with sleep off; if it stops,
   the same command resumes from the last `run\checkpoint-*` (one every 50 of 656 steps). Send back
   `$T\run\adapter` (~160 MB) and the last checkpoint's `trainer_state.json` (the loss curve).
4. Merge the adapter two ways, to compare them. Both outputs are unquantized (~15 GB each), as the GGUF converter needs:
   - `fused`: into the full-precision base, so the final quantization starts from clean weights;
   - `fused4`: into the 4-bit base it was trained on, so the weights are rounded to 4 bits twice.
   ```bash
   uv run mlx_lm.fuse --model "$ROOT/base" --adapter-path "$ROOT/adapters" --save-path "$ROOT/fused"
   uv run mlx_lm.fuse --model "$ROOT/base-4bit" --adapter-path "$ROOT/adapters" --save-path "$ROOT/fused4" --de-quantize
   ```
5. GGUF, then the same 4-bit level as today's model (`Q4_K_M`), for each merge:
   ```bash
   brew install llama.cpp
   git clone --depth 1 https://github.com/ggml-org/llama.cpp "$ROOT/llama.cpp"
   for m in fused fused4; do
     uv run python "$ROOT/llama.cpp/convert_hf_to_gguf.py" "$ROOT/$m" --outtype bf16 --outfile "$ROOT/$m-bf16.gguf"
     llama-quantize "$ROOT/$m-bf16.gguf" "$ROOT/$m-q4_k_m.gguf" Q4_K_M
   done
   ```
6. Register both in Ollama (each copies ~4.7 GB to the internal disk) and check the chat template came along:
   ```bash
   ollama create web-rag-llm-ft -f tuning/Modelfile
   sed 's/fused-q4_k_m/fused4-q4_k_m/' tuning/Modelfile > "$ROOT/Modelfile.ft4" && ollama create web-rag-llm-ft4 -f "$ROOT/Modelfile.ft4"
   ollama show web-rag-llm-ft --modelfile | grep -A3 TEMPLATE
   ```
7. Compare: judged `run_eval` on `golden.jsonl` and `golden_et.jsonl` with `LLM_MODEL` set to `web-rag-llm-ft`, then
   `web-rag-llm-ft4`, against the current `web-rag-llm` runs in the dashboard's Compare models page.
