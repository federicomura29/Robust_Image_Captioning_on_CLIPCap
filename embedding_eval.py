import torch
import clip
import json
from pydantic import BaseModel
from typing import List
from pathlib import Path
 
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model_type = "ViT-B/32"
 
# Load CLIP model and preprocessing (text only in this case)
model, preprocess = clip.load(clip_model_type, device=device, jit=False)
 
# Define paths
base_path = "Outputs/base_images_20250609_010349_captions.json"
noisy_path = "Outputs/noisy_images_20250609_010405_captions.json"
removed_path = "Outputs/removed_images_20250609_010429_captions.json"
 
# Load caption data
with open(base_path, "r") as f:
    base_data = json.load(f)
with open(noisy_path, "r") as f:
    noisy_data = json.load(f)
with open(removed_path, "r") as f:
    removed_data = json.load(f)
 
# Map image_id -> caption
base_dict = {item["image_id"]: item["caption"] for item in base_data}
noisy_dict = {item["image_id"]: item["caption"] for item in noisy_data}
removed_dict = {item["image_id"]: item["caption"] for item in removed_data}
 
# Use only common image IDs
all_ids = set(base_dict) & set(noisy_dict) & set(removed_dict)
 
# Class for holding caption triplets
class Caption(BaseModel):
    image_id: int
    base_caption: str
    noise_caption: str
    removed_caption: str
 
captions: List[Caption] = []
 
# Create triplets
for image_id in all_ids:
    captions.append(Caption(
        image_id=image_id,
        base_caption=base_dict[image_id],
        noise_caption=noisy_dict[image_id],
        removed_caption=removed_dict[image_id],
    ))
 
# Encode captions and calculate cosine similarity
for caption in captions:
    texts = [caption.base_caption, caption.noise_caption, caption.removed_caption]
    tokens = clip.tokenize(texts).to(device)  # Tokenize all 3 at once
 
    with torch.no_grad():
        embeddings = model.encode_text(tokens)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
 
    base_embed = embeddings[0]
    noise_embed = embeddings[1]
    removed_embed = embeddings[2]
 
    cosine_noise = (base_embed * noise_embed).sum().item()
    cosine_removed = (base_embed * removed_embed).sum().item()
 
    print(f"\nImage ID: {caption.image_id}")
    print(f"Base Caption: {caption.base_caption}")
    print(f"Noise Caption: {caption.noise_caption}")
    print(f"Removed Caption: {caption.removed_caption}")
    print(f"Cosine Similarity (Base vs Noise): {cosine_noise:.4f}")
    print(f"Cosine Similarity (Base vs Removed): {cosine_removed:.4f}")
 
