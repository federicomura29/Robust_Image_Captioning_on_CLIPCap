import os
import json
import torch
from PIL import Image
from tqdm import tqdm
from caption import generate_beam
from models.model import ClipCaptionPrefix
from transformers import GPT2Tokenizer
import clip


def get_device(device_id: int = 0) -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device(f"cuda:{device_id}")
    else:
        return torch.device("cpu")


DEVICE = get_device()
PREFIX_LENGTH = 40
REMOVED_DIR = "remove_output"
OUTPUT_DIR = "outputs"
REMOVAL_LOG = os.path.join(REMOVED_DIR, "removal_log.json")
MODEL_PATH = os.path.join("pretrained_models", "model_weights.pt")

# Load CLIP & CLIPCap
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
clip_model, preprocess = clip.load("RN50x4", device=DEVICE, jit=False)
clip_model = clip_model.eval().to(dtype=torch.float32, device=DEVICE)

model = ClipCaptionPrefix(
    prefix_length=PREFIX_LENGTH,
    clip_length=40,
    prefix_size=640,
    num_layers=8,
    mapping_type="transformer",
)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"), strict=False)
model = model.eval().to(DEVICE)


def generate_caption(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        prefix = clip_model.encode_image(image_tensor)
        prefix = prefix / prefix.norm(2, -1).item()
        prefix_embed = model.clip_project(prefix).reshape(1, PREFIX_LENGTH, -1)
        caption = generate_beam(model, tokenizer, embed=prefix_embed)[0]
    return caption.lower()


def main():
    with open(REMOVAL_LOG, "r") as f:
        removal_map = json.load(f)

    hallucination_results = {}

    for file_name, removed_objects in tqdm(removal_map.items()):
        image_id = file_name.split("_")[-1].split(".")[0]
        base_name = f"COCO_val2014_{image_id}.jpg"
        small_img = os.path.join(REMOVED_DIR, f"removed_small_{base_name}")
        large_img = os.path.join(REMOVED_DIR, f"removed_large_{base_name}")

        small_removed = removed_objects["small"].lower()
        large_removed = removed_objects["large"].lower()

        if not os.path.exists(small_img) or not os.path.exists(large_img):
            print(f"Missing image for ID {image_id}")
            continue

        small_caption = generate_caption(small_img)
        large_caption = generate_caption(large_img)

        hallucination_results[image_id] = {
            "small_removed": small_removed,
            "large_removed": large_removed,
            "small_caption": small_caption,
            "large_caption": large_caption,
            "hallucinated_small": small_removed in small_caption,
            "hallucinated_large": large_removed in large_caption,
        }

    out_path = os.path.join(OUTPUT_DIR, "hallucination_results.json")
    with open(out_path, "w") as f:
        json.dump(hallucination_results, f, indent=4)

    print(f"\n✅ Saved hallucination analysis to {out_path}")


if __name__ == "__main__":
    main()
