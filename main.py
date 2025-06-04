import numpy as np
import os
import torch
import clip
from Models.model import ClipCaptionPrefix
from transformers import GPT2Tokenizer
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import json

from Utils import (
    logger,
    img_processing,
    eval
)

from Utils.config import (
    DATA_VAL_PATH,
    DATA_EVAL_PATH,
    DEVICE,
    PRETRAINED_MODEL,
    EVAL,
    DATA_ANN_VAL_PATH,
)


if __name__ == "__main__":

    # Initialize logger
    logger.initialize_logger()
    logger = logger.get_logger()

    logger.info("-" * 50)
    logger.info("Executing main")
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
        img_processing.process_images(base_folder, "base", DATA_VAL_PATH, selected_images)
        logger.info("Base images processed and saved to %s", base_folder)

        # 2. Noisy images
        logger.info("Processing noisy images...")
        noise_folder = os.path.join(output_root, "noisy_images")
        img_processing.process_images(noise_folder, "noise", DATA_VAL_PATH, selected_images)
        logger.info("Noisy images processed and saved to %s", noise_folder)

        # 3. Object removal (assuming your friend implements this later)
        logger.info("Processing images for object removal...")
        removal_folder = os.path.join(output_root, "removed_images")
        img_processing.process_images(removal_folder, "removal", DATA_VAL_PATH, selected_images)
        logger.info("Object removal images processed and saved to %s", removal_folder)

    logger.info("-" * 50)
    logger.info("Loading model")
    logger.info("-" * 50)

    if EVAL:

        # Load the CLIP model and tokenizer
        CPU = torch.device("cpu")
        clip_model, preprocess = clip.load("RN50x4", device=DEVICE, jit=False)
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        logger.info("CLIP model and tokenizer loaded successfully")

        # Load the pre-trained model for captioning
        prefix_length = 40
        model = ClipCaptionPrefix(prefix_length, clip_length=40, prefix_size=640,
                                    num_layers=8, mapping_type='transformer')
        model.load_state_dict(torch.load(PRETRAINED_MODEL, map_location=CPU)) 
        model = model.eval() 
        model = model.to(DEVICE)
        logger.info("Pre-trained model loaded successfully")
        logger.info("Models loaded successfully")
        logger.info("-" * 50)

        # Generate captions for images
        logger.info("Generating captions for images in the evaluation folder...")

        logger.info("For base images:")
        # Generate captions for base images
        base_res, base_img_ids = eval.generate_captions(
            model=model,
            tokenizer=tokenizer,
            clip_model=clip_model,
            preprocess=preprocess,
            DATA_EVAL_PATH=DATA_EVAL_PATH,
            DEVICE=DEVICE,
            prefix_length=prefix_length,
            folder_name="base_images",
            use_beam_search=True
        )

        logger.info("For noisy images:")
        # Generate captions for noisy images
        noisy_res, noisy_img_ids = eval.generate_captions(
            model=model,
            tokenizer=tokenizer,
            clip_model=clip_model,
            preprocess=preprocess,
            DATA_EVAL_PATH=DATA_EVAL_PATH,
            DEVICE=DEVICE,
            prefix_length=prefix_length,
            folder_name="noisy_images",
            use_beam_search=True
        )

        """
        logger.info("For object removal images:")
        # Generate captions for object removal images
        rem_res, rem_img_ids =eval.generate_captions(
            model=model,
            tokenizer=tokenizer,
            clip_model=clip_model,
            preprocess=preprocess,
            DATA_EVAL_PATH=DATA_EVAL_PATH,
            DEVICE=DEVICE,
            prefix_length=prefix_length,
            folder_name="removed_images",
            use_beam_search=True
        )
        """
        logger.info("Caption generation completed for all images.")
        logger.info("-" * 50)

        logger.info("Compare to groundtruth captions and extract metrics...")

        base_metrics = eval.evaluate_captions(DATA_ANN_VAL_PATH,base_res, base_img_ids, "base_images")
        noisy_metrics = eval.evaluate_captions(DATA_ANN_VAL_PATH,noisy_res, noisy_img_ids, "noisy_images")
        """
        rem_metrics = eval.evaluate_captions(DATA_ANN_VAL_PATH,rem_res, rem_img_ids, "removed_images")
        """
        logger.info("Evaluation completed. Results saved in the evaluation folder.")
        logger.info("-" * 50)
        
