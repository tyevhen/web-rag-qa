# Training on a Windows PC with an NVIDIA GPU

PowerShell. Needs an NVIDIA driver 570 or newer (`nvidia-smi` lists the GPU), Git for Windows, ~30 GB free
disk (~70 GB with step 6b). Keep the PC plugged in with sleep off. In every new PowerShell window, run
`cd $HOME\web-rag-qa; $T = "$HOME\web-rag-tuning"` first.

**1. Get the code**
```powershell
cd $HOME
git clone -b qlora-tuning https://github.com/tyevhen/web-rag-qa.git
cd web-rag-qa
```
Already cloned: `cd $HOME\web-rag-qa; git fetch origin; git switch qlora-tuning; git pull`

**2. Install uv, then close and reopen PowerShell**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**3. Set up the environment**
```powershell
cd $HOME\web-rag-qa; $T = "$HOME\web-rag-tuning"
uv venv .venv-sft --python 3.13
uv pip install --python .venv-sft torch --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv-sft "trl==1.19.0" "peft==0.22.0" "bitsandbytes==0.50.2" "transformers==5.22.1" datasets
```

**4. Check the GPU** (must print `True` and the card)
```powershell
.venv-sft\Scripts\python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name())"
```

**5. Train** (the first ~10 min download the 15 GB model, then a `x/656` bar and a loss line every 10 steps)
```powershell
.venv-sft\Scripts\python tuning\colab_sft.py train --data tuning\data-4k --out $T\run
```
If it stops for any reason, run the same command again: it resumes from the last checkpoint (every 50 steps).
A loss around 1.0 that slowly falls is good; `nan` or `CUDA out of memory`: stop and send a screenshot.

**6a. Less than 32 GB of RAM: send the adapter** (~150 MB)
```powershell
Compress-Archive $T\run\adapter $HOME\Desktop\adapter.zip
```

**6b. 32 GB of RAM or more: merge, convert, quantize to 4-bit, send the GGUF** (~4.7 GB)

The file for the 16 GB Mac is `$T\fused-q4_k_m.gguf`: `Q4_K_M`, the same 4-bit level Ollama already uses (`llm/Modelfile`, ~4.7 GB). The bf16 GGUF is only the full-precision input to `llama-quantize`. Convert and quantize with one llama.cpp build — `winget list --id ggml.llamacpp` shows the installed tag; the converter checkout has to be that tag (`conversion/qwen.py` and the GGUF loader move between builds). This disk has about 17 GB free, so each intermediate is removed once the next file exists. Checkpoints only resume training; `run\adapter` stays.

```powershell
cd $HOME\Projects\web-rag-qa
$T = "$HOME\web-rag-tuning"
$run = Join-Path $T 'run'
$fused = Join-Path $T 'fused'
Remove-Item -Recurse -Force (Join-Path $run 'checkpoint-*')
.venv-sft\Scripts\python tuning\colab_sft.py merge --out $run --fused $fused
if ($LASTEXITCODE -ne 0) { throw "merge failed" }
Remove-Item -Recurse -Force $HOME\.cache\huggingface\hub\models--Qwen--Qwen2.5-7B-Instruct
winget upgrade --id ggml.llamacpp -e --accept-package-agreements --accept-source-agreements
```
Open a new PowerShell (so the upgraded `llama-quantize` is on `PATH`), then:
```powershell
cd $HOME\Projects\web-rag-qa
$T = "$HOME\web-rag-tuning"
$fused = Join-Path $T 'fused'
$convert = Join-Path $T 'llama.cpp\convert_hf_to_gguf.py'
$bf16 = Join-Path $T 'fused-bf16.gguf'
$q4 = Join-Path $T 'fused-q4_k_m.gguf'
$ver = ((winget list --id ggml.llamacpp --disable-interactivity | Select-String -Pattern '\bb\d+\b').Matches.Value | Select-Object -First 1)
if (-not (Test-Path (Join-Path $T 'llama.cpp\.git'))) {
  git clone --depth 1 --branch $ver https://github.com/ggml-org/llama.cpp (Join-Path $T 'llama.cpp')
} else {
  git -C (Join-Path $T 'llama.cpp') fetch --depth 1 origin "refs/tags/${ver}:refs/tags/${ver}"
  git -C (Join-Path $T 'llama.cpp') checkout $ver
}
if (-not (Test-Path .venv-gguf\Scripts\python.exe)) { uv venv .venv-gguf --python 3.12 }
uv pip install --python .venv-gguf -r (Join-Path $T 'llama.cpp\requirements\requirements-convert_hf_to_gguf.txt')
.venv-gguf\Scripts\python $convert $fused --outtype bf16 --outfile $bf16
if ($LASTEXITCODE -ne 0) { throw "convert failed" }
Remove-Item -Recurse -Force $fused
llama-quantize $bf16 $q4 Q4_K_M
if ($LASTEXITCODE -ne 0) { throw "quantize failed; kept $bf16" }
Remove-Item $bf16
```
Send `$T\fused-q4_k_m.gguf` by a Google Drive or OneDrive link (git and free WeTransfer won't take it). On the Mac: `ollama create web-rag-llm-ft -f tuning/Modelfile` after that file is at the `FROM` path in `tuning/Modelfile`.

**7. Validation loss, if the training log is gone** (optional; ~10 min, plus the 15 GB base download if 6b removed it)
```powershell
cd $HOME\Projects\web-rag-qa; $T = "$HOME\web-rag-tuning"
.venv-sft\Scripts\python tuning\colab_sft.py eval --data tuning\data-4k --out (Join-Path $T 'run')
```
Needs only `run\adapter`. Prints two lines, `base:` (the model without the adapter) and `tuned:`, each with
`eval_loss` and token accuracy on `valid.jsonl`; send a screenshot. The tuned loss should be well below the base's.
Delete `$HOME\.cache\huggingface` afterwards to get the disk back.
