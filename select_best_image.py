import os
import argparse

# local imports
from evaluate_images import calculate_image_score

# parsing arguments
parser = argparse.ArgumentParser()
parser.add_argument('--image_name', type=str, default="girl_with_kitty")
parser.add_argument('--prompt', type=str, default="a girl with a kitty")
parser.add_argument('--number_images', type=int)
parser.add_argument('--show_image', action='store_true')
parser.add_argument('--save_dir', type=str, default='images') # path to saved generated images
args = parser.parse_args()


# transform arguments
image_name = args.image_name.replace(' ','_')
images_dir = args.save_dir+'/'+image_name


# evaluate and show generated images
calculate_image_score(images_dir, args.prompt, args.number_images, args.show_image)


