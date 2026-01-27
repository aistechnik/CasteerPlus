import os
import argparse
import torch
import pickle
import subprocess

# local imports
from controller import VectorStore, register_vector_control
from models import get_model
from evaluate_images import calculate_image_score

# parsing arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, choices=['sdxl', 'sdxl-turbo', 'sdxl-tuned', 'sdxl-turbo-image'], default="sdxl-turbo")
parser.add_argument('--image_name', type=str, default="girl_with_kitty")
parser.add_argument('--prompt', type=str, default="a girl with a kitty")
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--steering_vectors', type=str) # path to steering vectors file
parser.add_argument('--steer_only_up', action='store_true')
parser.add_argument('--num_denoising_steps', type=int, default=30) # 30 for sdxl, 1 for turbo, 2 for turbo-image
parser.add_argument('--steer_back', action='store_true')
parser.add_argument('--alpha', type=str, default="10")
parser.add_argument('--beta', type=int, default=2)
parser.add_argument('--evaluate_images', action='store_true')
parser.add_argument('--save_dir', type=str, default='casteer_images') # path to saving generated images
args = parser.parse_args()


pipe = get_model(args.model)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
pipe.to(device)
pipe.set_progress_bar_config(disable=True)

def run_model(model_type, pipe, prompt, seed, num_denoising_steps):
    if args.model in ['sdxl', 'sdxl-tuned']:
        image = pipe(prompt=prompt,
                     num_inference_steps=num_denoising_steps,
                     generator=torch.Generator(device=device).manual_seed(seed)
                    ).images[0]

    elif args.model in ['sdxl-turbo']:
        image = pipe(prompt=prompt,
                     num_inference_steps=num_denoising_steps,
                     guidance_scale=0.0,
                     generator=torch.Generator(device=device).manual_seed(seed)
                    ).images[0]

    elif args.model in ['sdxl-turbo-image']:
        init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png").resize((512, 512))

        image = pipe(prompt=prompt,image=init_image
                     num_inference_steps=2, # num_denoising_steps,
                     strength=0.5,
                     guidance_scale=0.0,
                     generator=torch.Generator(device=device).manual_seed(seed)
                    ).images[0]

    return image


# transform arguments
image_name = args.image_name.replace(' ','_')
image_save_dir = args.save_dir+'/'+image_name
alphas = args.alpha.split(',')
number_images = len(alphas)
if args.evaluate_images and number_images > 1:
    evaluate_images = True
else:
    evaluate_images = False

if not os.path.exists(image_save_dir):
    os.makedirs(image_save_dir)

with open(args.steering_vectors, 'rb') as handle:
    steering_vectors = pickle.load(handle)

print('Generating for prompt:')
print(args.prompt)

for i in range(len(alphas)):
    alphai = int(alphas[i])
    print('Step with alpha:', alphai)
    if alphai == 0:
        is_origin_image = True
    ##
    controller = VectorStore(steering_vectors, device=device)
    controller.steer_only_up = True if args.steer_only_up else False

    if args.steer_back:
        controller.steer_back = True
        controller.beta = args.beta
    else:
        controller.steer_back = False
        controller.alpha = alphai

    register_vector_control(pipe.unet, controller)

    image = run_model(args.model, pipe, args.prompt, args.seed, args.num_denoising_steps)
    image.save(os.path.join(image_save_dir, "{}.png".format(str(i+1))))

# evaluate generated images
if evaluate_images:
    calculate_image_score(image_save_dir, args.prompt, number_images, False)
