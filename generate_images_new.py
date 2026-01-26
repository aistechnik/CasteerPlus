import os
import argparse
import subprocess

# local imports
from controller import VectorStore, register_vector_control
from models import get_model
from sel_best_image import get_score_image

# parsing arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, choices=['sdxl', 'sdxl-turbo', 'sdxl-tuned', 'sdxl-turbo-tuned'], default="sdxl-turbo")
parser.add_argument('--image_name', type=str, default="girl_with_kitty")
parser.add_argument('--prompt', type=str, default="a girl with a kitty")
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--steering_vectors', type=str) # path to steering vectors file
parser.add_argument('--steer_only_up', action='store_true')
parser.add_argument('--num_denoising_steps', type=int, default=50) # 1 for turbo, 30 for sdxl
parser.add_argument('--steer_back', action='store_true')
parser.add_argument('--alpha', type=str, default="10")
parser.add_argument('--beta', type=int, default=2)
parser.add_argument('--save_dir', type=str, default='casteer_images') # path to saving generated images
parser.add_argument('--select_best_image', action='store_true')
args = parser.parse_args()


# transform arguments
image_name = args.image_name.replace(' ','_')
image_save_dir = args.save_dir+'/'+image_name
alphas = args.alpha.split(',')
number_images = len(alphas)


if args.select_best_image:
    get_score_image(image_save_dir, args.prompt, number_images)
