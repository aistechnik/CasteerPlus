import os
import pickle
import torch

# local imports
import image_utils
import steering
import prompt_catalog
from models import get_model

# parsing arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, choices=['sdxl', 'sdxl-turbo', 'sdxl-tuned'], default="sdxl-turbo")
parser.add_argument('--mode', type=str, choices=['concrete', 'human-related', 'style'], default="style")
parser.add_argument('--image_name', type=str, default="girl_with_kitty")
parser.add_argument('--prompt', type=str, default="a girl with a kitty")
parser.add_argument('--num_denoising_steps', type=int, default=50) #  1 for turbo, 30 for sdxl
parser.add_argument('--concept_pos', type=str, default="anime")
parser.add_argument('--concept_neg', type=str, default="classic")
parser.add_argument('--save_dir', type=str, default='steering_vectors') # path to saving steering vectors
parser.add_argument('--image_save_dir', type=str, default='images') # path to saving generated images
args = parser.parse_args()

pipe = get_model(args.model)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
pipe.to(device)
pipe.set_progress_bar_config(disable=True)



## Hyper Paramters
STEER_TYPE = "default"
INF_STEPS = 20
GUIDE_SCALE = 5.0
# SCALE_STEER = 1.0 # not applied during cache creation

# Loads "calibration dataset": dataset from which steering vectors are derived from
if args.mode == "style":
    prompts = prompt_catalog.ANIME_PROMPT[:20]
elif args.mode == "metal":
    prompts = prompt_catalog.METALLIC_SCULPTURE_SET[:20]
elif args.mode == "fine_image":
    prompts = prompt_catalog.FINE_IMAGE_PROMPT[:20]

else:
  raise NotImplementedError(f"Steering prompt mode {args.mode} not implemented")

# add hooks to collect activations and later applies steering vector to intermiate activations
steer_hooks = steering.add_steer_hooks(pipe, steer_type=STEER_TYPE, save_every=1)

steering_vectors = steering.build_final_steering_vectors(
    pipe,
    steer_hooks,
    prompts,
    num_inference_steps=INF_STEPS,
    guidance_scale=GUIDE_SCALE
)
print("save_dir, len", args.save_dir, len(steering_vectors))
# Saving steering vectors:
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
with open(os.path.join(args.save_dir, '{}_{}_{}.pickle'.format(args.model, args.concept_pos, args.concept_neg)), 'wb') as handle:
    pickle.dump(steering_vectors, handle)

print("steering_vectors shape: ", steering_vectors[0].shape) # (20, 2, 640)

# adds calculated steering vectors to hooks so it can be applied during forward pass
steering.add_final_steer_vectors(steer_hooks, steering_vectors)

TEST_PROMPTS = [ args.prompt ]
print(TEST_PROMPTS)
# generates images using steering vectors
STEER_SCALE_LIST = [0.0, 1.0, 2.0, 10.0]
steering.run_grid_experiment(
    pipe, steer_hooks, TEST_PROMPTS,
    num_inference_steps=INF_STEPS,
    steer_type=STEER_TYPE,
    gscale_list=[GUIDE_SCALE],
    steer_scale_list=STEER_SCALE_LIST, # (0.0: no steering (baseline), 10.0: strong steering)
    out_root=f"{args.mode}_experiments",
)


# Example usage:
folder = f"{args.mode}_experiments"
save_dir = os.path.join(folder, args.image_save_dir)

# Generate and save all comparison plots
saved = image_utils.plot_all_param_images(folder, save_dir=save_dir)

# List what was saved
available = image_utils.list_saved_plots(save_dir)
path = available[0]

print(path)

# Load one saved plot
if available:
    image_utils.load_saved_plot(path, save_dir=save_dir)


