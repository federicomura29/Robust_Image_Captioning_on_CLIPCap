import os
import random
from PIL import Image, ImageFilter, ImageDraw
from skimage import io
import numpy as np
import json
from collections import defaultdict
import torch
from skimage.morphology import binary_dilation
from diffusers import StableDiffusionInpaintPipeline


from Utils.logger import get_logger
from Utils.config import (
    DATA_VAL_PATH,
    DATA_EVAL_PATH,
)

logger = get_logger()

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
    
'''
Input:

image_path: straightforward, the path to the image object is to be removed from.
captions_path: should look like --> "annotations/captions_train2014.json"
instances_path: should look like --> "annotations/instances_train2014.json"
selection_mode: what type of strategy to use to pick the object to be removed;
                "random"(default): picks a random object from the image
                "smallest": picks the smallest object
                "largest": picks the largest object
                "first": picks the first object in the metadata
                

Output:

Returns a tuple (image_original, image_removed, removal_info)

- image_original: the original image loaded from `image_path` as a PIL Image.
- image_removed: the modified image where the selected object has been inpainted (removed), PIL Image.
- removal_info: a dictionary containing metadata about the removal process, including:
    - 'caption': the image caption used to guide inpainting
    - 'object_category': the category name of the removed object (e.g., "person", "dog")
    - 'selection_mode': the selection strategy used (e.g., "random", "largest")
    - 'bbox': the bounding box of the removed object in COCO format [x, y, width, height]
    
Note: If an error shows up about python types or similar, try doing this: pip install "numpy<2" --force-reinstall
'''
def remove_object(image_path, captions_path, instances_path, selection_mode="random"):
    
    # --- Load COCO data ---
    with open(captions_path, "r") as f:
        captions_data = json.load(f)
    with open(instances_path, "r") as f:
        instances_data = json.load(f)

    # --- Build mapping dicts ---
    image_id_to_filename = {img["id"]: img["file_name"] for img in instances_data["images"]}
    category_id_to_name = {cat["id"]: cat["name"] for cat in instances_data["categories"]}
    annotations_by_image = defaultdict(list)
    captions_by_image = defaultdict(list)

    for ann in instances_data["annotations"]:
        annotations_by_image[ann["image_id"]].append(ann)
    for cap in captions_data["annotations"]:
        captions_by_image[cap["image_id"]].append(cap["caption"])

    # --- Get image filename and ID ---
    image_filename = os.path.basename(image_path)
    image_id = next((img["id"] for img in instances_data["images"] if img["file_name"] == image_filename), None)
    if image_id is None:
        raise ValueError("Image not found in COCO metadata")

    # --- Select object ---
    objects = annotations_by_image[image_id]
    if not objects:
        raise ValueError("No objects found for this image")

    if selection_mode == "first":
        obj = objects[0]
    elif selection_mode == "random":
        obj = random.choice(objects)
    elif selection_mode == "largest":
        obj = max(objects, key=lambda o: o["bbox"][2] * o["bbox"][3])
    elif selection_mode == "smallest":
        obj = min(objects, key=lambda o: o["bbox"][2] * o["bbox"][3])
    else:
        raise ValueError("Invalid selection_mode")

    # --- Mask creation ---
    def segmentation_to_mask(segmentation, image_size):
        mask = Image.new('L', image_size, 0)
        for polygon in segmentation:
            polygon = [tuple(polygon[i:i+2]) for i in range(0, len(polygon), 2)]
            ImageDraw.Draw(mask).polygon(polygon, outline=1, fill=1)
        return np.array(mask)

    original_image = Image.open(image_path).convert("RGB")
    width, height = original_image.size
    mask_np = segmentation_to_mask(obj["segmentation"], (width, height))

    # --- Dilate mask ---
    dilated = binary_dilation((mask_np > 0).astype(np.uint8), iterations=25).astype(np.uint8)
    mask_pil = Image.fromarray((dilated * 255).astype(np.uint8)).convert("L")

    # --- Resize for Stable Diffusion ---
    original_resized = original_image.resize((512, 512)).convert("RGB")
    mask_resized = mask_pil.resize((512, 512)).convert("L")

    # --- Inpaint using Stable Diffusion ---
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16
    ).to("cuda")
    pipe.safety_checker = lambda images, **kwargs: (images, False)

    result = pipe(
        prompt="the same background texture extended to fill the removed area",
        image=original_resized,
        mask_image=mask_resized
    ).images[0]

    # --- Metadata ---
    removal_info = {
        "image_id": image_id,
        "image_filename": image_filename,
        "removed_object": category_id_to_name[obj["category_id"]],
        "bbox": obj["bbox"],
        "original_captions": captions_by_image[image_id],
        "new_caption": None
    }

    return original_resized, result, removal_info

def process_images(new_folder, config, source_folder, selected_images, captions_path="", instances_path=""):
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
            original_img, processed_img, metadata = remove_object(image_path,captions_path,instances_path,"random")
        elif config == 'base':
            processed_img = Image.open(image_path)
        else:
            raise ValueError("Invalid config. Use 'noise', 'removal', or 'base'.")

        # Save processed image
        save_path = os.path.join(new_folder, filename)
        processed_img.save(save_path)


logger.initialize_logger()
logger = logger.get_logger()

logger.info("-" * 50)
logger.info("Executing image processing script")
logger.info("-" * 50)

if not os.path.exists(DATA_EVAL_PATH):
    
    os.makedirs(DATA_EVAL_PATH)
    logger.info("Created evaluation directory: %s", DATA_EVAL_PATH)
    logger.info("Loading images from COCO validation set...")
    all_images = [f for f in os.listdir(DATA_VAL_PATH) if f.lower().endswith('.jpg')]
    selected_images = all_images[:100]  # Select 100 consistent images

    # === Run the processing with different configurations ===
    output_root = DATA_EVAL_PATH
        
    # 1. Base images (no modification)
    logger.info("Processing base images...")
    base_folder = os.path.join(output_root, "base_images")
    process_images(base_folder, "base", DATA_VAL_PATH, selected_images)
    logger.info("Base images processed and saved to %s", base_folder)

    # 2. Noisy images
    logger.info("Processing noisy images...")
    noise_folder = os.path.join(output_root, "noisy_images")
    process_images(noise_folder, "noise", DATA_VAL_PATH, selected_images)
    logger.info("Noisy images processed and saved to %s", noise_folder)

    # 3. Object removal (assuming your friend implements this later)
    logger.info("Processing images for object removal...")
    removal_folder = os.path.join(output_root, "removed_images")
    process_images(removal_folder, "removal", DATA_VAL_PATH, selected_images)
    logger.info("Object removal images processed and saved to %s", removal_folder)

logger.info("-" * 50)
logger.info("Image processing completed successfully")
logger.info("-" * 50)
