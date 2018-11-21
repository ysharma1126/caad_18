#!/bin/bash
#
# run_defense.sh is a script which executes the defense
#
# Environment which runs attacks and defences calls it in a following way:
#   run_defense.sh INPUT_DIR OUTPUT_FILE
# where:
#   INPUT_DIR - directory with input PNG images
#   OUTPUT_FILE - file to store classification labels
#
# Checkpoints are available at https://github.com/tensorflow/models/tree/master/slim
# and https://github.com/tensorflow/models/tree/master/research/adv_imagenet_models

INPUT_DIR=$1
OUTPUT_FILE=$2

TMP_DIR=/tmp
mkdir -p ${TMP_DIR}/noise-added

start=`date +%s`

python randomize.py \
  --input_dir="${INPUT_DIR}" \
  --output_dir="${TMP_DIR}/noise-added" 

python defense_incres.py \
  --input_dir="${TMP_DIR}/noise-added" \
  --output_file="${TMP_DIR}/pred_5.csv" \
  --checkpoint_path=adv_inception_resnet_v2.ckpt

python defense_v3.py \
  --input_dir="${TMP_DIR}/noise-added" \
  --output_file="${TMP_DIR}/pred_4.csv" \
  --checkpoint_path=adv_inception_v3.ckpt

python defense_incres.py \
  --input_dir="${TMP_DIR}/noise-added" \
  --output_file="${TMP_DIR}/pred_1.csv" \
  --checkpoint_path=ens_adv_inception_resnet_v2.ckpt

python defense_v3.py \
  --input_dir="${TMP_DIR}/noise-added" \
  --output_file="${TMP_DIR}/pred_2.csv" \
  --checkpoint_path=ens4_adv_inception_v3.ckpt

python defense_v3.py \
  --input_dir="${TMP_DIR}/noise-added" \
  --output_file="${TMP_DIR}/pred_3.csv" \
  --checkpoint_path=ens3_adv_inception_v3.ckpt

python defense_merge.py \
  --tmp_dir="${TMP_DIR}" \
  --output_file="${OUTPUT_FILE}"
end=`date +%s`

runtime=$((end-start))

echo $runtime
