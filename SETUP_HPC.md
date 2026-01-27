# HPC Setup Guide for Modular LoRA Training

## Pre-download Requirements

### ✅ What you MUST download before HPC (with internet):

1. **Stable Diffusion Model** - Run this on your local machine:

```bash
python download_models.py \
    --model_id stable-diffusion-v1-5/stable-diffusion-v1-5 \
    --save_full_pipeline \
    --pipeline_dir ./sd_model
```

This creates a folder `sd_model/` with all model files (~5GB).

2. **Transfer to HPC:**
```bash
# Copy the model to HPC
scp -r sd_model btfarre2@login.hpc.ncsu.edu:/share/rkmeente/btfarre2/model/stable_diffusion_v1_5
```

### ✅ What gets cached automatically (first run only):

These will download automatically on first run if HPC has internet access:
- Tokenizer vocabulary files
- Model configs (small JSON files)

They'll be cached in `/share/rkmeente/btfarre2/hf_cache/` thanks to the environment variables in the .sh script.

## Offline Mode Setup

If HPC has NO internet, you need to pre-cache everything:

```bash
# On local machine with internet
python download_models.py \
    --model_id stable-diffusion-v1-5/stable-diffusion-v1-5 \
    --cache_dir ./hf_cache
```

Then copy `hf_cache/` to HPC:
```bash
scp -r hf_cache btfarre2@login.hpc.ncsu.edu:/share/rkmeente/btfarre2/
```

## Installation on HPC

Your conda environment needs these packages:

```bash
pip install torch torchvision torchaudio
pip install diffusers transformers
pip install peft datasets pillow tqdm
```

**Note:** You do NOT need `accelerate` or `bitsandbytes` for this modular approach.

## File Structure on HPC

```
/home/btfarre2/gsv_host_detector/tree_classification/
├── modular/                                    # Copy entire folder
│   ├── __main__.py
│   ├── train_lora.py
│   ├── train_text_to_image_lora.py
│   ├── prep_dataset.py
│   ├── generate_images.py
│   └── download_models.py
├── modular_config.json                         # Config file
└── sd_modular.sh                               # Job script

/share/rkmeente/btfarre2/
├── model/
│   └── stable_diffusion_v1_5/                  # Pre-downloaded model
├── datasets/
│   └── prepared_lora/                          # Prepared datasets with metadata.jsonl
│       ├── acer/
│       │   ├── acer_00001.jpg
│       │   ├── acer_00002.jpg
│       │   └── metadata.jsonl
│       ├── fraxinus/
│       └── ...
├── stablediffusion-modular/                    # Output directory
└── hf_cache/                                   # HuggingFace cache
```

## Step-by-Step Workflow

### 1. Prepare datasets (on HPC or local)

```bash
python -m modular.prep_dataset \
    --input_base /path/to/raw/images \
    --output_base /share/rkmeente/btfarre2/datasets/prepared_lora \
    --genera acer fraxinus quercus \
    --max_images 1000
```

### 2. Update config file

Edit `modular_config.json` with your paths and genera list.

### 3. Submit job

```bash
cd /home/btfarre2/gsv_host_detector/tree_classification
bsub < sd_modular.sh
```

### 4. Monitor progress

```bash
tail -f stdout.*
tail -f stderr.*
```

## What Gets Downloaded (if internet available)

On first run, these small files auto-download:
- `tokenizer/vocab.json` (~500KB)
- `tokenizer/merges.txt` (~500KB)
- Config JSONs (~10KB each)

Total: ~1-2MB

## Fully Offline Mode

If HPC has zero internet:

1. On local machine, run:
```bash
python download_models.py \
    --model_id stable-diffusion-v1-5/stable-diffusion-v1-5 \
    --cache_dir ./full_cache \
    --save_full_pipeline \
    --pipeline_dir ./sd_model
```

2. Copy both to HPC:
```bash
scp -r sd_model btfarre2@login.hpc.ncsu.edu:/share/rkmeente/btfarre2/model/stable_diffusion_v1_5
scp -r full_cache btfarre2@login.hpc.ncsu.edu:/share/rkmeente/btfarre2/hf_cache
```

3. The environment variables in `sd_modular.sh` will use the cache:
```bash
export HF_HOME=/share/rkmeente/btfarre2/hf_cache
export TRANSFORMERS_CACHE=/share/rkmeente/btfarre2/hf_cache
```

## Checking What's Needed

To see exactly what will be downloaded:

```bash
# Dry run - shows what would download
python -c "
from transformers import CLIPTokenizer
from diffusers import AutoencoderKL
print('This will download if not cached...')
tokenizer = CLIPTokenizer.from_pretrained('stable-diffusion-v1-5/stable-diffusion-v1-5', subfolder='tokenizer')
print('Tokenizer OK')
"
```

## Summary

**Minimum to pre-download:**
- Stable Diffusion model files (~5GB) ✅ REQUIRED

**Optional (auto-downloads if internet available):**
- Tokenizer files (~1MB) - Usually available
- Config files (~10KB) - Usually available

**Best practice:** Pre-download everything to be safe!
