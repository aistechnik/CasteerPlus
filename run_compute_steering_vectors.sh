#!/bin/bash

python3 compute_steering_vectors.py --model="sdxl-turbo" --mode="style" --concept_pos="anime" --concept_neg="classic" --num_denoising_steps=20 --save_dir="/content/drive/My Drive/steering_vectors"
