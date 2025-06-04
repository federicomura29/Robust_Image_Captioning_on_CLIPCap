# ...existing code...

from PIL import Image
from tqdm import tqdm
from Utils.logger import get_logger
from Utils.caption_pred import generate_beam, generate2
import os
import torch
from datetime import datetime
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import json
from Utils.config import (
    OUTPUT_PATH,
)


logger = get_logger()

def generate_captions(model, tokenizer, clip_model, preprocess, DATA_EVAL_PATH, DEVICE, prefix_length=77, folder_name="base_images",use_beam_search=True):

    """
    Generate captions for images in the specified evaluation folder.

    Parameters:
        model: The caption generation model.
        tokenizer: The tokenizer for the model.
        clip_model: The CLIP model for image encoding.
        preprocess: Preprocessing function for images.
        DATA_EVAL_PATH: Path to the evaluation data folder.
        DEVICE: Device to run the model on (e.g., 'cuda' or 'cpu').
        prefix_length: Length of the prefix for caption generation.
        folder_name: Name of the folder containing images to evaluate.
        use_beam_search: Whether to use beam search for caption generation.

    Returns:
        List of dictionaries with image IDs and generated captions.
    """

    # Choose which folder to process (e.g., "base_images")
    eval_folder = os.path.join(DATA_EVAL_PATH, folder_name)
    image_files = [f for f in os.listdir(eval_folder) if f.lower().endswith('.jpg')]
    images_path = [os.path.join(eval_folder, f) for f in image_files]

    results = []
    image_ids = []

    for idx, img_path in tqdm(enumerate(images_path), total=len(images_path)):
        pil_image = Image.open(img_path).convert("RGB")
        image_tensor = preprocess(pil_image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            prefix = clip_model.encode_image(image_tensor).to(DEVICE, dtype=torch.float32)
            prefix = prefix / prefix.norm(2, -1).item()
            prefix_embed = model.clip_project(prefix).reshape(1, prefix_length, -1)

        if use_beam_search:
            generated_text_prefix = generate_beam(model, tokenizer, embed=prefix_embed)[0]
        else:
            generated_text_prefix = generate2(model, tokenizer, embed=prefix_embed)

        logger.info(f"Generated caption for image {idx + 1}/{len(images_path)}: {generated_text_prefix}")

        # Prepare the result dictionary with the image_id and generated caption
        file_name = os.path.basename(images_path[idx])
        image_id = int(file_name.split("_")[-1].split(".")[0])  # Extract the integer ID
        result = {
            "image_id": image_id,
            "caption": generated_text_prefix
        }

        image_ids.append(image_id)
        results.append(result)

    # Save results to a json file with the folder name and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_file = os.path.join(OUTPUT_PATH, f"{folder_name}_{timestamp}_captions.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"Captions saved to {output_file}")

    return results, image_ids


def evaluate_captions(data_eval_path, resFile, image_ids, type="base_images"):

    logger.info("Starting evaluation of generated captions from %s", type)

    coco = COCO(data_eval_path)

    # Load your results into COCO result object
    cocoRes = coco.loadRes(resFile)

    # Filter COCO annotations to the images you evaluated
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = image_ids

    # Run evaluation
    cocoEval.evaluate()

    # Print evaluation scores and adding them in Output folder
    output_eval_file = os.path.join(OUTPUT_PATH, f"{type}_metrics.json")
    with open(output_eval_file, 'w') as f:
        for metric, score in cocoEval.eval.items():
            logger.info(f'{metric}: {score:.4f}')
            f.write(f'{metric}: {score:.4f}\n')
    logger.info("Metric results saved to %s", output_eval_file)
    logger.info("Evaluation completed successfully.")

    return cocoEval.eval

