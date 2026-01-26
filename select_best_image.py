import os
import argparse
import torch
import ImageReward as RM
from PIL import Image


# parsing arguments
parser = argparse.ArgumentParser()
parser.add_argument('--image_name', type=str, default="girl_with_kitty")
parser.add_argument('--prompt', type=str, default="a girl with a kitty")
parser.add_argument('--show_image', action='store_true')
parser.add_argument('--save_dir', type=str, default='images') # path to saving generated images
args = parser.parse_args()


# transform arguments
image_name = args.image_name.replace(' ','_')
img_prefix = args.save_dir+'/'+image_name
print(img_prefix)
print(args.prompt)

generations = [f"{pic_id}.png" for pic_id in range(1, 5)]
img_list = [os.path.join(img_prefix, img) for img in generations]
model = RM.load("ImageReward-v1.0")

with torch.no_grad():
    ranking, rewards = model.inference_rank(args.prompt, img_list)
    # Print the result
    print("\nPreference predictions:")
    # print(f"ranking = {ranking}")
    # print(f"rewards = {rewards}")
    best_score = 0.0
    best_index = 0
    for index in range(len(img_list)):
        score = model.score(args.prompt, img_list[index])
        print(f"{generations[index]:>8s}: {score:.4f}")
        if score > best_score:
            best_score = score
            best_index = index

print(f"The best image with the highest ImageReward score is {img_list[best_index]}")

if args.show_image:
    image = Image.open(img_list[best_index])
    image.show()


