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
    eval
)

from Utils.config import (
    DATA_VAL_PATH,
    DATA_EVAL_PATH,
    DEVICE,
    PRETRAINED_MODEL,
    EVAL,
    DATA_ANN_PATH,
)


if __name__ == "__main__":

    # Initialize logger
    logger.initialize_logger()
    logger = logger.get_logger()

    logger.info("-" * 50)
    logger.info("Executing main")
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
        
        logger.info("Caption generation completed for all images.")
        logger.info("-" * 50)

        logger.info("Compare to groundtruth captions and extract metrics...")

        captions_path = os.path.join(DATA_ANN_PATH, "captions_val2014.json")
        base_metrics = eval.evaluate_captions(captions_path,base_res, base_img_ids, "base_images")
        noisy_metrics = eval.evaluate_captions(captions_path,noisy_res, noisy_img_ids, "noisy_images")
        rem_metrics = eval.evaluate_captions(captions_path,rem_res, rem_img_ids, "removed_images")
        logger.info("Evaluation completed. Results saved in the evaluation folder.")
        logger.info("-" * 50)
        
