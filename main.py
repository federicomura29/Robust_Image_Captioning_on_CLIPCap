import os
import torch
import clip
from models.model import ClipCaptionPrefix
from transformers import GPT2Tokenizer
from eval import evaluate_captions, generate_captions
import logging
import ssl
import certifi

# ssl._create_default_https_context = ssl.create_default_context  # <- ensure this is a callable
# ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

CPU = torch.device("cpu")


def get_device(device_id: int) -> torch.device:
    if not torch.cuda.is_available():
        return CPU
    device_id = min(torch.cuda.device_count() - 1, device_id)
    return torch.device(f"cuda:{device_id}")


CUDA = get_device

model_path = os.path.join("pretrained_models", "model_weights.pt")


device = "cpu"
clip_model, preprocess = clip.load("RN50x4", device=device, jit=False)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# SPECIAL_TOKENS = {
    # "additional_special_tokens": ["<SELF_ANALYSIS>", "<REVISED_CAPTION>", "<END>"]
# }
# tokenizer.add_special_tokens(SPECIAL_TOKENS)

prefix_length = 40 
model = ClipCaptionPrefix(
    prefix_length=prefix_length,
    clip_length=40,
    prefix_size=640,
    num_layers=8,
    mapping_type="transformer",
)

# model.gpt.resize_token_embeddings(len(tokenizer))
model.load_state_dict(torch.load(model_path, map_location=CPU), strict=False)


model = model.eval()
model = model.to(device)

DATA_EVAL_PATH = "eval_images"

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
        use_beam_search=True,
    )

    annotations_to_compare = "captions_val2014.json"
    evaluate_captions(annotations_to_compare, result_file, image_ids, type=folder)
