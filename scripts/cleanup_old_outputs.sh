#!/bin/bash
#BSUB -n 1
#BSUB -W 10
#BSUB -J cleanup_old_outputs
#BSUB -o stdout.%J
#BSUB -e stderr.%J
#BSUB -q debug
#BSUB -R rusage[mem=2]

# Deletes lora_weights/ and synth3.5/ from each genus folder
# under synthetic/autoarborist/ and synthetic/inat/

DATA_DIR="/rs1/researchers/c/cmjone25/auto_arborist_cvpr2022_v0.15/data/synthetic"

for dataset in autoarborist inat; do
    dir="$DATA_DIR/$dataset"
    if [ ! -d "$dir" ]; then
        echo "Skipping $dataset (not found)"
        continue
    fi
    for genus in "$dir"/*/; do
        for target in lora_weights synth3.5; do
            if [ -d "$genus$target" ]; then
                echo "Removing $genus$target"
                rm -rf "$genus$target"
            fi
        done
    done
done

echo "Done."
