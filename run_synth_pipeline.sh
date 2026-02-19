#!/bin/bash
set -e
#BSUB -n 4
#BSUB -W 4320
#BSUB -J synth_lora_pipeline
#BSUB -o stdout.%J
#BSUB -e stderr.%J
#BSUB -q gpu
#BSUB -R span[hosts=1]
#BSUB -gpu "num=1:mode=exclusive_process:mps=no"
#BSUB -R "select[ h100 || a100 || l40 || l40s ]"
#BSUB -R rusage[mem=64]

module load conda
source activate /usr/local/usrapps/rkmeente/btfarre2/conda_envs/sd_lora_torch260

export HF_DATASETS_CACHE=/share/rkmeente/btfarre2/model/model_cache/datasets
export TRANSFORMERS_CACHE=/share/rkmeente/btfarre2/model/model_cache
export HF_HOME=/share/rkmeente/btfarre2/model/model_cache
export MODEL_CACHE=/share/rkmeente/btfarre2/model/model_cache
export TMPDIR=/share/rkmeente/btfarre2/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export PATH=/usr/local/usrapps/rkmeente/btfarre2/conda_envs/sd_lora_torch260/bin:$PATH

SYNTH_ROOT="/rs1/researchers/c/cmjone25/auto_arborist_cvpr2022_v0.15/data/synthetic"
SCRIPT_DIR="/home/btfarre2/gsv_host_detector/tree_classification/sd-repo/sd3.5-with-lora"

echo "=========================================="
echo "LoRA Pipeline"
echo "=========================================="
echo "Synth root: $SYNTH_ROOT"
echo "Script dir: $SCRIPT_DIR"
echo ""

cd "$SCRIPT_DIR" || exit 1

# Train LoRA models
echo "Training LoRA adapters"
python train/train_synth.py \
    --synth_root "$SYNTH_ROOT" \
    --config modular_config.json

echo ""
echo "Generating synthetic images"
python generate/generate_synth.py \
    --synth_root "$SYNTH_ROOT" \
    --config modular_config.json

echo ""
echo "=========================================="
echo "Pipeline complete!"
echo "=========================================="

conda deactivate
