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

**6b. 32 GB of RAM or more: merge, convert, send the GGUF** (~4.7 GB)
```powershell
.venv-sft\Scripts\python tuning\colab_sft.py merge --out $T\run --fused $T\fused
winget install llama.cpp
```
Open a new PowerShell, then:
```powershell
cd $HOME\web-rag-qa; $T = "$HOME\web-rag-tuning"
git clone --depth 1 https://github.com/ggml-org/llama.cpp $T\llama.cpp
uv venv .venv-gguf --python 3.12
uv pip install --python .venv-gguf -r $T\llama.cpp\requirements\requirements-convert_hf_to_gguf.txt
.venv-gguf\Scripts\python $T\llama.cpp\convert_hf_to_gguf.py $T\fused --outtype bf16 --outfile $T\fused-bf16.gguf
llama-quantize $T\fused-bf16.gguf $T\fused-q4_k_m.gguf Q4_K_M
```
Send `$T\fused-q4_k_m.gguf` by a Google Drive or OneDrive link (git and free WeTransfer won't take it).
