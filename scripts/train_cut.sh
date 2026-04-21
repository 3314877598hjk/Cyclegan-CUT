#!/usr/bin/env bash
set -e

DATASET=${1:-./datasets/maps}

# CUT + PatchNCE + Sobel edge loss + attention.
python train.py \
  --dataroot "${DATASET}" \
  --name map2vector_cut \
  --model cut \
  --CUT_mode CUT \
  --direction AtoB \
  --lambda_edge 1.0 \
  --n_epochs 100 \
  --n_epochs_decay 100

# FastCUT keeps the one-way backend and uses the lighter official-style mode.
python train.py \
  --dataroot "${DATASET}" \
  --name map2vector_fastcut \
  --model cut \
  --CUT_mode FastCUT \
  --direction AtoB \
  --lambda_edge 1.0 \
  --n_epochs 100 \
  --n_epochs_decay 100
