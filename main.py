import os
import torch
import clip
from models.model import ClipCaptionPrefix
from transformers import GPT2Tokenizer
from eval import evaluate_captions, generate_captions
import platform
import logging

logging.basicConfig(level=logging.INFO)

logging.info(platform.processor())
logging.info(torch.__version__)
logging.info(f"{torch.backends.mps.is_available()} is MPS available")


CPU = torch.device("cpu")


def get_device(device_id: int = 0) -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device(f"cuda:{device_id}")
    else:
        return torch.device("cpu")


model_path = os.path.join("pretrained_models", "model_weights.pt")


device = get_device()
clip_model, preprocess = clip.load("RN50x4", device=device, jit=False)
clip_model = clip_model.to(dtype=torch.float32)
clip_model = clip_model.to(device)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

prefix_length = 40
model = ClipCaptionPrefix(
    prefix_length,
    clip_length=40,
    prefix_size=640,
    num_layers=8,
    mapping_type="transformer",
)

state_dict = torch.load(model_path, map_location=CPU)
model.load_state_dict(state_dict, strict=False)
model = model.eval().to(device)

DATA_EVAL_PATH = "eval_images/less_images_for_testing"

for folder in ["base_images", "noisy_images", "removed_images"]:
    results, image_ids, result_file = generate_captions(
        model,
        tokenizer,
        clip_model,
        preprocess,
        DATA_EVAL_PATH,
        device,
        prefix_length=prefix_length,
        folder_name=folder,
    )

    annotations_to_compare = "captions_val2014.json"
    evaluate_captions(annotations_to_compare, result_file, image_ids, type=folder)
