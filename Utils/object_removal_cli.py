import os
import json
import random
from collections import defaultdict
from PIL import Image, ImageDraw
import numpy as np
from scipy.ndimage import binary_dilation
from diffusers import StableDiffusionInpaintPipeline
import torch
import matplotlib.pyplot as plt
torch.cuda.empty_cache()

# ----------------------------- Configuration -----------------------------
image_folder = "/home/jovyan/Desktop/ACVPR-Project/data/coco/val2014"
output_folder = "/home/jovyan/Desktop/ACVPR-Project/remove_output"
captions_path = "/home/jovyan/Desktop/ACVPR-Project/data/coco/annotations/captions_val2014.json"
instances_path = "/home/jovyan/Desktop/ACVPR-Project/data/coco/annotations/instances_val2014.json"
log_path = "/home/jovyan/Desktop/ACVPR-Project/remove_output/removal_log.json"

# Load existing log if it exists
if os.path.exists(log_path):
    with open(log_path, "r") as f:
        removal_log = json.load(f)
else:
    removal_log = {}

# ----------------------------- Metadata Loading -----------------------------
print("🔄 Loading metadata...")

with open(captions_path, "r") as f:
    captions_data = json.load(f)
with open(instances_path, "r") as f:
    instances_data = json.load(f)

image_id_to_filename = {img["id"]: img["file_name"] for img in instances_data["images"]}
filename_to_image_id = {v: k for k, v in image_id_to_filename.items()}
category_id_to_name = {cat["id"]: cat["name"] for cat in instances_data["categories"]}
annotations_by_image = defaultdict(list)
captions_by_image = defaultdict(list)

for ann in instances_data["annotations"]:
    annotations_by_image[ann["image_id"]].append(ann)
for cap in captions_data["annotations"]:
    captions_by_image[cap["image_id"]].append(cap["caption"])

image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(".jpg")])
random.shuffle(image_files)

# ----------------------------- Load Inpainting Model -----------------------------
print("🎨 Loading inpainting model...")
device = torch.device("cuda:1")  # Set your GPU here
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16
).to(device)

pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))

# ----------------------------- Mask Utility -----------------------------
def segmentation_to_mask(segmentation, image_size):
    mask = Image.new('L', image_size, 0)
    for seg in segmentation:
        if isinstance(seg[0], (int, float)):
            points = [tuple(seg[i:i+2]) for i in range(0, len(seg), 2)]
        else:
            points = []
            for subseg in seg:
                points.extend([tuple(subseg[i:i+2]) for i in range(0, len(subseg), 2)])
        if points:
            ImageDraw.Draw(mask).polygon(points, outline=1, fill=1)
    return np.array(mask)

# ----------------------------- Inpainting Function -----------------------------
def inpaint_and_save(img, obj, filename, label):
    mask_np = segmentation_to_mask(obj["segmentation"], img.size)
    dilated = binary_dilation((mask_np > 0).astype(np.uint8), iterations=15).astype(np.uint8)
    mask_pil = Image.fromarray((dilated * 255).astype(np.uint8)).convert("L")

    img_resized = img.resize((512, 512)).convert("RGB")
    mask_resized = mask_pil.resize((512, 512)).convert("L")

    print(f"🧠 Inpainting ({label})...")
    result = pipe(
        prompt="realistic background continuation matching the image texture",
        image=img_resized,
        mask_image=mask_resized
    ).images[0]

    result_original_size = result.resize(img.size, Image.LANCZOS)

    plt.figure(figsize=(8, 6))
    plt.imshow(result_original_size)
    plt.axis('off')
    plt.title(f"Removed: {category_id_to_name[obj['category_id']]} ({label})")
    plt.show()

    output_path = os.path.join(output_folder, f"{filename[:-4]}_removed_{label}.jpg")
    result_original_size.save(output_path)
    print(f"✅ Saved to {output_path}")

# ----------------------------- Main Loop -----------------------------
for filename in image_files:
    image_path = os.path.join(image_folder, filename)
    image_id = filename_to_image_id.get(filename)

    if image_id is None or filename in removal_log:
        continue

    annotations = annotations_by_image[image_id]
    if not annotations:
        continue

    print(f"\n🖼️ Image: {filename}")
    img = Image.open(image_path).convert("RGB")
    img.show()

    print("💬 Captions:")
    for cap in captions_by_image[image_id]:
        print(f" - {cap}")

    print("\n📦 Objects:")
    for i, obj in enumerate(annotations):
        cat_name = category_id_to_name[obj["category_id"]]
        bbox = obj["bbox"]
        area = round(bbox[2] * bbox[3])
        print(f"[{i}] {cat_name:<15} → BBox Area: {area}")

    selected = {}
    for label in ["small", "large"]:
        while True:
            user_input = input(f"👉 Select index of {label} object to remove (or -1 to skip image, 'q' to quit): ").strip().lower()

            if user_input in ['q', 'quit']:
                print("👋 Exiting the loop.")
                exit(0)
            if user_input == "-1":
                print("⏩ Skipping image.")
                selected = None
                break
            try:
                idx = int(user_input)
                if idx < 0 or idx >= len(annotations):
                    raise IndexError
                selected[label] = annotations[idx]
                break
            except (ValueError, IndexError):
                print("❌ Invalid input. Enter a valid index, -1 to skip, or 'q' to quit.")

        if selected is None:
            break  # exit inner loop early

    if selected is None:
        continue  # skip the image entirely

    # ----- Inpaint both but do not save yet -----
    results = {}

    for label in ["small", "large"]:
        print(f"\n🧠 Inpainting {label} object...")
        obj = selected[label]

        # Mask generation
        mask_np = segmentation_to_mask(obj["segmentation"], img.size)
        dilated = binary_dilation((mask_np > 0).astype(np.uint8), iterations=5).astype(np.uint8)
        mask_pil = Image.fromarray((dilated * 255).astype(np.uint8)).convert("L")

        # Resize
        img_resized = img.resize((512, 512)).convert("RGB")
        mask_resized = mask_pil.resize((512, 512)).convert("L")

        # Inpaint
        result = pipe(
            prompt="realistic background continuation matching the image texture",
            image=img_resized,
            mask_image=mask_resized
        ).images[0]

        # Resize back to original
        result_original_size = result.resize(img.size, Image.LANCZOS)

        # Show result
        plt.figure(figsize=(8, 6))
        plt.imshow(result_original_size)
        plt.axis('off')
        plt.title(f"{label.capitalize()} object removed: {category_id_to_name[obj['category_id']]}")
        plt.show()

        results[label] = {
            "image": result_original_size,
            "object": category_id_to_name[obj["category_id"]],
            "index": annotations.index(obj)
        }

    # ----- Final confirmation -----
    confirm = input("💾 Do you want to keep and save the results? (y/n): ").strip().lower()
    if confirm == "y":
        for label, data in results.items():
            output_path = os.path.join(output_folder, f"removed_{label}_{filename}")
            data["image"].save(output_path)
            print(f"✅ Saved {label} image to {output_path}")

        removal_log[filename] = {
            "small": results["small"]["object"],
            "large": results["large"]["object"]
        }

        with open(log_path, "w") as f:
            json.dump(removal_log, f, indent=2)
        print("📝 Log updated.")
    else:
        print("❌ Discarded inpainted results for this image.")


