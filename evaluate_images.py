import os
import argparse
import torch
import ImageReward as RM

def calculate_image_score(img_prefix, prompt, num_images, is_origin_image):
    prompt = prompt + ", anime style"
    print("is_origin_image",is_origin_image)
    #size = num_images + 1
    indx = 0 if is_origin_image else 1
    size = num_images if is_origin_image else num_images + 1
    generations = [f"{pic_id}.png" for pic_id in range(indx, size)]
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

