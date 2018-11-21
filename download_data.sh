#!/bin/bash

cd "$( dirname "${BASH_SOURCE[0]}" )"

# Download checkpoints for sample attacks and defenses.
nontarget_attack/download_checkpoints.sh
target_attack/download_checkpoints.sh
defenses/2_MSB/download_checkpoints.sh
defenses/Dropout/download_checkpoints.sh

# Download dataset.
mkdir -p dataset/images
python ./dataset/download_images.py \
  --input_file=dataset/dev_dataset.csv \
  --output_dir=dataset/images/
