from pydantic import BaseModel
from typing import List
import json
import clip
import os
import numpy as np
import torch

base_path = "outputs/base_images_20250609_010349_captions.json"
noisy_path = "outputs/noisy_images_20250609_010405_captions.json"
removed_path = "outputs/removed_images_20250609_010429_captions.json"


class Caption(BaseModel):
    image_id: int
    base_caption: str
    noise_caption: str
    removed_caption: str
    similarity_noise: float = 0.0
    similarity_removed: float = 0.0


with open(base_path, "r") as f:
    base_data = json.load(f)

with open(noisy_path, "r") as f:
    noisy_data = json.load(f)

with open(removed_path, "r") as f:
    removed_data = json.load(f)


base_dict = {item["image_id"]: item["caption"] for item in base_data}
noisy_dict = {item["image_id"]: item["caption"] for item in noisy_data}
removed_dict = {item["image_id"]: item["caption"] for item in removed_data}


all_ids = set(base_dict) & set(noisy_dict) & set(removed_dict)

captions: List[Caption] = []

for image_id in all_ids:
    triplet = Caption(
        image_id=image_id,
        base_caption=base_dict[image_id],
        noise_caption=noisy_dict[image_id],
        removed_caption=removed_dict[image_id],
    )
    captions.append(triplet)

model, preprocess = clip.load("RN50x4", device="cpu")

for caption in captions:
    base_embedding = model.encode_text(clip.tokenize(caption.base_caption).to("cpu")).squeeze(0)
    noise_embedding = model.encode_text(clip.tokenize(caption.noise_caption).to("cpu")).squeeze(0)
    removed_embedding = model.encode_text(clip.tokenize(caption.removed_caption).to("cpu")).squeeze(0)

    similarity_noise = torch.nn.functional.cosine_similarity(base_embedding, noise_embedding, dim=0)
    similarity_removed = torch.nn.functional.cosine_similarity(base_embedding, removed_embedding, dim=0)
    caption.similarity_noise = similarity_noise.item()
    caption.similarity_removed = similarity_removed.item()

# for caption in captions:
#     print(f"Image ID: {caption.image_id}")
#     print(f"Base Caption: {caption.base_caption}")
#     print(f"Noise Caption: {caption.noise_caption} (Similarity: {caption.similarity_noise:.4f})")
#     print(f"Removed Caption: {caption.removed_caption} (Similarity: {caption.similarity_removed:.4f})")
#     print("-" * 40)

# interesting_noisy_images = [c for c in captions if c.similarity_noise < 0.8]
# interesting_removed_images = [c for c in captions if c.similarity_removed > 0.8]

# interesting_noisy_images = sorted(interesting_noisy_images, key=lambda x: x.similarity_noise, reverse=True)
# interesting_removed_images = sorted(interesting_removed_images, key=lambda x: x.similarity_removed)

# print(len(interesting_noisy_images), "interesting_noisy_images found")
# print(len(interesting_removed_images), "interesting_removed_images  found")

output_path = "outputs/images_removed.json"
# save only removed captions, dont save noisy captions,
captions_to_save = [
    {
        "image_id": caption.image_id,
        "base_caption": caption.base_caption,
        "removed_caption": caption.removed_caption,
        "similarity_removed": caption.similarity_removed,
    }
    for caption in captions
]

captions_to_save = sorted(captions_to_save, key=lambda x: x["similarity_removed"], reverse=True)


with open(output_path, "w") as f:
    json.dump(
        {
            "images": captions_to_save,
        },
        f,
        indent=4,
    )

output_path = "outputs/images_noisy.json"
# save only noisy captions, dont save removed captions,
captions_to_save = [
    {
        "image_id": caption.image_id,
        "base_caption": caption.base_caption,
        "noise_caption": caption.noise_caption,
        "similarity_noise": caption.similarity_noise,
    }
    for caption in captions
]

captions_to_save = sorted(captions_to_save, key=lambda x: x["similarity_noise"])
with open(output_path, "w") as f:
    json.dump(
        {
            "images": captions_to_save,
        },
        f,
        indent=4,
    )
