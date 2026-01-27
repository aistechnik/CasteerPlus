import os
import argparse
import torch
import ImageReward as RM
from PIL import Image

def calculate_image_score(img_prefix, prompt, num_images, show_image):
    prompt = prompt + ", anime style"
    size = num_images + 1

    generations = [f"{pic_id}.png" for pic_id in range(1, size)]
    img_list = [os.path.join(img_prefix, img) for img in generations]
    model = RM.load("ImageReward-v1.0")

    with torch.no_grad():
        ranking, rewards = model.inference_rank(prompt, img_list)
        # Print the result
        print("\nPreference predictions score:")
        # print(f"ranking = {ranking}")
        # print(f"rewards = {rewards}")
        best_score = -100.0
        best_index = 0
        for index in range(len(img_list)):
            score = model.score(prompt, img_list[index]) * 10
            print(f"{generations[index]:>8s}: {score:.4f}")
            if score > best_score:
                best_score = score
                best_index = index

    print(f"The best image with the highest score is {img_list[best_index]}")

    if show_image:
        image = Image.open(img_list[best_index])
        image.show()
