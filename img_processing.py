import os
import random
from PIL import Image, ImageFilter
from skimage import io
import numpy as np

def add_noise(image_path, noise_type='gaussian'):
    """
    Apply Gaussian blur or salt-and-pepper noise to an image.

    Parameters:
        image_path (str): Path to the image.
        noise_type (str): 'gaussian' or 's&p'

    Returns:
        PIL.Image: Image with noise applied.
    """
    image = io.imread(image_path)
    pil_image = Image.fromarray(image)

    if noise_type == 'gaussian':
        return pil_image.filter(ImageFilter.GaussianBlur(radius=2))
    else:
        raise ValueError("Unsupported noise_type. Use 'gaussian'.")
    

def remove_object(image_path):
    """
    Placeholder for object removal logic.
    """
    pass


def process_images(new_folder, config, source_folder, selected_images):
    """
    Process images with noise or object removal and save them to a new folder.

    Parameters:
        new_folder (str): Name of the output folder.
        config (str): 'noise', 'removal', or 'base'.
        source_folder (str): Folder where original images are stored.
        selected_images (List[str]): List of filenames to process.
    """
    os.makedirs(new_folder, exist_ok=True)

    for filename in selected_images:
        image_path = os.path.join(source_folder, filename)

        if config == 'noise':
            processed_img = add_noise(image_path, noise_type='gaussian')
        elif config == 'removal':
            processed_img = remove_object(image_path)  # expected to return a PIL image
        elif config == 'base':
            processed_img = Image.open(image_path)
        else:
            raise ValueError("Invalid config. Use 'noise', 'removal', or 'base'.")

        # Save processed image
        save_path = os.path.join(new_folder, filename)
        processed_img.save(save_path)


if __name__ == "__main__":

    val_images_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/coco/val_2014")
    all_images = [f for f in os.listdir(val_images_dir) if f.lower().endswith('.jpg')]
    selected_images = all_images[:100]  # Select 100 consistent images

    # === Run the processing with different configurations ===
    output_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_images")
    

    # 1. Base images (no modification)
    base_folder = os.path.join(output_root, "base_images")
    process_images(base_folder, "base", val_images_dir, selected_images)

    # 2. Noisy images
    noise_folder = os.path.join(output_root, "noisy_images")
    process_images(noise_folder, "noise", val_images_dir, selected_images)

    # 3. Object removal (assuming your friend implements this later)
    removal_folder = os.path.join(output_root, "removed_images")
    process_images(removal_folder, "removal", val_images_dir, selected_images)