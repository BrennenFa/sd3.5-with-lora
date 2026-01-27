# Modular LoRA Training Pipeline

This folder contains a modular approach to LoRA fine-tuning for Stable Diffusion, based on Tom's approach but adapted for your workflow.

## Files

1. **download_models.py** - Download models for offline HPC use
2. **prep_dataset.py** - Prepare datasets with metadata.jsonl
3. **train_text_to_image_lora.py** - Core training script (similar to HuggingFace's official script)
4. **train_lora.py** - Orchestrator script to train multiple genera
5. **generate_images.py** - Generate images using trained LoRA models

## Workflow

### Step 1: Download models (on machine with internet)

```bash
# Download to cache
python download_models.py \
    --model_id stable-diffusion-v1-5/stable-diffusion-v1-5 \
    --cache_dir ./model_cache \
    --save_full_pipeline \
    --pipeline_dir ./sd_model
```

Transfer `model_cache` and/or `sd_model` to HPC.

### Step 2: Prepare datasets

```bash
python prep_dataset.py \
    --input_base /path/to/raw/images \
    --output_base /path/to/prepared/datasets \
    --genera acer fraxinus quercus \
    --max_images 1000
```

### Step 3: Create config file

Create `config.json`:

```json
{
    "experiment": "tree_lora",
    "model_path": "/path/to/sd_model",
    "train_data_dir": "/path/to/prepared/datasets",
    "output_path": "/path/to/outputs",
    "selected_genera": ["acer", "fraxinus", "quercus"],
    "resolution": 512,
    "train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "epochs": 10,
    "learning_rate": 1e-4,
    "mixed_precision": "fp16",
    "checkpointing_steps": 500,
    "validation_epochs": 1,
    "use_wandb": false
}
```

### Step 4: Train (on HPC)

On HPC, set environment variables:

```bash
export HF_HOME=/path/to/model_cache
export TRANSFORMERS_CACHE=/path/to/model_cache
```

Then run training:

```bash
python train_lora.py --config config.json
```

### Step 5: Generate images

```bash
python generate_images.py \
    --config config.json \
    --num_images 12 \
    --guidance_scale 7.5 \
    --num_inference_steps 30
```

## Key Differences from Original Approach

### Tom's approach (original):
- Uses external HuggingFace training script via subprocess
- Re-encodes images every epoch
- Simpler, less code

### This modular approach:
- Self-contained training script (no external dependencies)
- Still re-encodes every batch (like HuggingFace approach)
- Uses accelerate for distributed training
- Easier to customize and debug
- Works offline once models are downloaded

## Offline HPC Usage

1. Download models on local machine with internet
2. Copy `model_cache` or `sd_model` to HPC
3. Update config.json to point to local model path
4. Set `HF_HOME` environment variable
5. Run training completely offline

## Requirements

```bash
pip install torch torchvision torchaudio
pip install diffusers transformers accelerate
pip install peft bitsandbytes
pip install datasets pillow tqdm
```
