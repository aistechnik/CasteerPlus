#!/bin/bash

python3 -m casteer.compute_steering_vectors --model="sdxl-turbo" --mode="style" --concept_pos="anime" --concept_neg="classic" --num_denoising_steps=20 --save_dir="/content/drive/My Drive/steering_vectors"
