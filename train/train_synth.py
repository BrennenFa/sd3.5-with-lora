#!/usr/bin/env python3
"""
Train LoRA adapters for all genera in synthetic dataset.
Automatically discovers genera from directory structure and trains each one.

Usage:
    python train_synth.py --synth_root /path/to/synth --config modular_config.json
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path


def discover_genera_and_datasets(synth_root):
    """
    Discover datasets and genera from directory structure.

    Expected structure:
        synth_root/
        ├── inat/{genus}/lora/
        │   └── (100 training images)
        └── autoarborist/{genus}/lora/
            └── (100 training images)

    Returns: dict of {dataset: [list of genera]}
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
        genera = []

        # Discover genera for this dataset
        for genus_dir in sorted(dataset_dir.iterdir()):
            if not genus_dir.is_dir():
                continue

            lora_path = genus_dir / "lora"
            if lora_path.exists() and lora_path.is_dir():
                # Check if there are images
                image_files = list(lora_path.glob("*.jpg")) + \
                             list(lora_path.glob("*.jpeg")) + \
                             list(lora_path.glob("*.png")) + \
                             list(lora_path.glob("*.webp"))

                if image_files:
                    genera.append(genus_dir.name)

        if genera:
            structure[dataset_name] = genera
            print(f"Found dataset '{dataset_name}' with {len(genera)} genera")

    return structure if structure else None


def train_genus(
    genus,
    train_data_dir,
    output_dir,
    config,
):
    """
    Train LoRA for a single genus.

    Args:
        genus: Genus name
        train_data_dir: Directory containing training images
        output_dir: Output directory for checkpoints
        config: Config dictionary with training parameters
    """
    print(f"\n{'='*60}")
    print(f"TRAINING LORA FOR: {genus}")
    print(f"{'='*60}\n")

    if not os.path.exists(train_data_dir):
        print(f"ERROR: training folder not found at {train_data_dir}")
        return False

    os.makedirs(output_dir, exist_ok=True)

    # Path to training script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_script = os.path.join(script_dir, "train_text_to_image_lora.py")

    # Build command from config
    cmd = [
        sys.executable,
        train_script,
        f"--pretrained_model_name_or_path={config['model_path']}",
        f"--train_data_dir={train_data_dir}",
        "--dataloader_num_workers=4",
        f"--resolution={config.get('resolution', 1024)}",
        "--center_crop",
        "--random_flip",
        f"--train_batch_size={config.get('train_batch_size', 1)}",
        f"--gradient_accumulation_steps={config.get('gradient_accumulation_steps', 4)}",
        f"--num_train_epochs={config.get('epochs', 10)}",
        f"--learning_rate={config.get('learning_rate', 1e-4)}",
        "--lr_scheduler=constant",
        "--lr_warmup_steps=0",
        "--seed=42",
        f"--output_dir={output_dir}",
        f"--checkpointing_steps={config.get('checkpointing_steps', 500)}",
        "--checkpoints_total_limit=2",
        f"--mixed_precision={config.get('mixed_precision', 'fp16')}",
        "--report_to=none",
    ]

    if config.get('enable_xformers'):
        cmd.append("--enable_xformers")

    print("Launching training with command:")
    print(" ".join(cmd), "\n")

    try:
        subprocess.run(cmd, check=True)
        print(f"\n✓ Training completed for {genus}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Training failed for {genus}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Train LoRA adapters for all genera in synthetic dataset"
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

    # Discover genera and datasets
    print(f"\nDiscovering datasets and genera in {args.synth_root}")
    structure = discover_genera_and_datasets(args.synth_root)

    if not structure:
        print("No datasets found!")
        return 1

    total_genera = sum(len(genera) for genera in structure.values())
    print(f"\nFound {len(structure)} datasets with {total_genera} total genera\n")

    # Train each genus in each dataset
    results = {}
    synth_path = Path(args.synth_root)

    for dataset_name in sorted(structure.keys()):
        print(f"\n{'='*60}")
        print(f"PROCESSING DATASET: {dataset_name}")
        print(f"{'='*60}")

        dataset_path = synth_path / dataset_name
        results[dataset_name] = {}

        for genus in structure[dataset_name]:
            genus_path = dataset_path / genus
            train_data_dir = genus_path / "lora"
            lora_weights_dir = genus_path / "lora_weights"

            # Check if already trained
            if (lora_weights_dir / "adapter_config.json").exists():
                print(f"\n{genus}: Already trained, skipping...")
                results[dataset_name][genus] = "skipped"
                continue

            # Train
            success = train_genus(
                genus=genus,
                train_data_dir=str(train_data_dir),
                output_dir=str(lora_weights_dir),
                config=config,
            )

            results[dataset_name][genus] = "success" if success else "failed"

    # Print summary
    print(f"\n\n{'='*60}")
    print("TRAINING SUMMARY")
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
