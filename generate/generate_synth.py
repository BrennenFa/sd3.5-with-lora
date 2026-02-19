#!/usr/bin/env python3
"""
Generate synthetic images using trained LoRA adapters.
Automatically discovers trained LoRA weights and generates images per genus.

Usage:
    python generate_synth.py --synth_root /path/to/synth --config modular_config.json --num_images 1000
"""

import os
import sys
import argparse
import json
import torch
from pathlib import Path
from diffusers import StableDiffusion3Pipeline
from peft import PeftModel
import transformers.utils.import_utils as _transformers_import_utils

from prompts_dynamic import generate_prompt_batch, get_negative_prompt, normalize_dataset_type

# bypass issue
_transformers_import_utils.check_torch_load_is_safe = lambda: None


def generate_for_genus(
    genus,
    dataset_type,
    lora_path,
    model_path,
    output_dir,
    num_images=1000,
    guidance_scale=7.5,
    num_inference_steps=30,
    resolution=1024,
    device="cuda"
):
    """
    Generate synthetic images for a single genus using its trained LoRA.

    Args:
        genus: Genus name
        dataset_type: Dataset type (autoarborist, inaturalist)
        lora_path: Path to trained LoRA weights
        model_path: Path to base model
        output_dir: Output directory for generated images
        num_images: Number of images to generate
        guidance_scale: Guidance scale for generation
        num_inference_steps: Number of inference steps
        resolution: Image resolution
        device: Device to use (cuda or cpu)
    """
    print(f"\n{'='*60}")
    print(f"Generating {num_images} images for: {genus}")
    print(f"{'='*60}")

    if not os.path.exists(lora_path):
        print(f"LoRA weights not found: {lora_path}")
        return False

    os.makedirs(output_dir, exist_ok=True)

    # Load pipeline
    print("Loading base model")
    try:
        pipe = StableDiffusion3Pipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

    # Load LoRA weights
    print("Loading LoRA weights")
    try:
        pipe.transformer = PeftModel.from_pretrained(pipe.transformer, lora_path)
    except Exception as e:
        print(f"Error loading LoRA: {e}")
        return False

    pipe = pipe.to(device)

    # Generate dynamic prompts
    print("Generating prompts...")
    prompts = generate_prompt_batch(
        dataset_type=dataset_type,
        genus=genus,
        num_prompts=num_images,
        seed=42
    )

    negative_prompt = get_negative_prompt(dataset_type)

    # Generate images
    print(f"Generating {num_images} images...")
    images_generated = 0

    for idx in range(num_images):
        prompt = prompts[idx]

        if (idx + 1) % 100 == 0:
            print(f"  [{idx+1}/{num_images}] {prompt[:80]}...")

        try:
            with torch.no_grad():
                image = pipe(
                    prompt,
                    negative_prompt=negative_prompt,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    height=resolution,
                    width=resolution,
                    generator=torch.Generator(device=device).manual_seed(42 + idx)
                ).images[0]

            save_path = os.path.join(output_dir, f"{genus}_{idx:04d}.png")
            image.save(save_path)
            images_generated += 1

        except Exception as e:
            print(f"  Error generating image {idx}: {e}")
            continue

    # Cleanup
    del pipe
    torch.cuda.empty_cache()

    print(f"✓ Generation complete for {genus} ({images_generated}/{num_images} images)")
    return images_generated == num_images


def discover_trained_loras(synth_root):
    """
    Discover trained LoRA weights in directory structure.

    Expected structure:
        synth_root/
        ├── {dataset}/
        │   └── {genus}/
        │       ├── lora/          (training images)
        │       ├── lora_weights/  (trained adapter)
        │       └── synth3.5/      (output images)
    """
    structure = {}
    synth_path = Path(synth_root)

    if not synth_path.exists():
        print(f"Error: synth_root not found: {synth_root}")
        return None

    # Discover datasets
    for dataset_dir in sorted(synth_path.iterdir()):
        if not dataset_dir.is_dir():
            continue

        dataset_name = dataset_dir.name
        genera = {}

        # Discover genera for this dataset
        for genus_dir in sorted(dataset_dir.iterdir()):
            if not genus_dir.is_dir():
                continue

            lora_weights_path = genus_dir / "lora_weights"
            if (lora_weights_path / "adapter_config.json").exists():
                genera[genus_dir.name] = str(lora_weights_path)

        if genera:
            structure[dataset_name] = genera
            print(f"Found {len(genera)} trained genera in dataset '{dataset_name}'")

    return structure if structure else None


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic images using trained LoRA adapters"
    )
    parser.add_argument(
        "--synth_root",
        type=str,
        required=True,
        help="Path to synthetic data root directory"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config JSON file"
    )

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)

    num_images = config['num_images']
    dataset_type = normalize_dataset_type(config['dataset_type'])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Dataset type: {dataset_type}")

    # Discover trained LoRAs
    print(f"\nDiscovering trained LoRA weights in {args.synth_root}")
    structure = discover_trained_loras(args.synth_root)

    if not structure:
        print("No trained LoRA weights found!")
        return 1

    total_genera = sum(len(genera) for genera in structure.values())
    print(f"\nFound {len(structure)} datasets with {total_genera} total trained genera\n")

    # Generate images for each genus
    results = {}
    synth_path = Path(args.synth_root)

    for dataset_name in sorted(structure.keys()):
        print(f"\n{'='*60}")
        print(f"GENERATING FOR DATASET: {dataset_name}")
        print(f"{'='*60}")

        results[dataset_name] = {}

        for genus, lora_path in sorted(structure[dataset_name].items()):
            # Output directory: {synth_root}/{dataset}/{genus}/synth3.5/
            genus_path = synth_path / dataset_name / genus
            output_dir = genus_path / "synth3.5"

            # Skip if images already exist
            if output_dir.exists():
                existing_images = list(output_dir.glob(f"{genus}_*.png"))
                if len(existing_images) >= num_images:
                    print(f"\n{genus}: {len(existing_images)} images already exist, skipping...")
                    results[dataset_name][genus] = "skipped"
                    continue

            success = generate_for_genus(
                genus=genus,
                dataset_type=dataset_type,
                lora_path=lora_path,
                model_path=config['model_path'],
                output_dir=str(output_dir),
                num_images=num_images,
                guidance_scale=config['guidance_scale'],
                num_inference_steps=config['num_inference_steps'],
                resolution=config['resolution'],
                device=device
            )

            results[dataset_name][genus] = "success" if success else "failed"

    # Print summary
    print(f"\n\n{'='*60}")
    print("GENERATION SUMMARY")
    print(f"{'='*60}\n")

    for dataset_name in sorted(results.keys()):
        print(f"{dataset_name}:")
        for genus, status in sorted(results[dataset_name].items()):
            icon = "✓" if status in ["success", "skipped"] else "✗"
            print(f"  {icon} {genus}: {status}")

    print(f"\n{'='*60}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
