import os
import torch
import clip
from models.model import ClipCaptionPrefix
from transformers import GPT2Tokenizer
from eval import evaluate_captions, generate_captions, evaluate_single_caption
import platform
import logging
import json
from pycocotools.coco import COCO


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

ANNOTATION_FILE = "captions_val2014.json"


def eval_all():
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

        evaluate_captions(ANNOTATION_FILE, result_file, image_ids, type=folder)


def load_captions(file_path):
    with open(file_path, "r") as f:
        return {item["image_id"]: item["caption"] for item in json.load(f)}


def eval_single():
    OUTPUT_PATH = "outputs"
    base_captions = load_captions(
        os.path.join(
            OUTPUT_PATH, "baseline/baseline_base_images_20250617_234459_captions.json"
        )
    )
    noisy_captions = load_captions(
        os.path.join(
            OUTPUT_PATH, "baseline/baseline_noisy_images_20250617_234735_captions.json"
        )
    )

    coco = COCO(ANNOTATION_FILE)
    all_metrics = {}

    for image_id in base_captions:
        if image_id not in noisy_captions:
            continue

        base_eval = evaluate_single_caption(coco, image_id, base_captions[image_id])
        noisy_eval = evaluate_single_caption(coco, image_id, noisy_captions[image_id])

        metric_diff = {
            metric: noisy_eval[metric] - base_eval[metric] for metric in base_eval
        }

        all_metrics[image_id] = {
            "base_caption": base_captions[image_id],
            "noisy_caption": noisy_captions[image_id],
            "base_metrics": base_eval,
            "noisy_metrics": noisy_eval,
            "metric_diff": metric_diff,
        }

    output_file = os.path.join(OUTPUT_PATH, "imagewise_metric_diff.json")
    with open(output_file, "w") as f:
        json.dump(all_metrics, f, indent=4)
    logging.info(f"Saved image-wise metric differences to {output_file}")


def find_improved_noise_images():
    input_path = "outputs/imagewise_metric_diff.json"
    output_path = "outputs/imagewise_metric_diff_that_improved_after_blur.json"

    with open(input_path, "r") as f:
        data = json.load(f)

    improved_data = {
        image_id: details
        for image_id, details in data.items()
        if all(diff >= 0 for diff in details["metric_diff"].values())
        and any(diff > 0 for diff in details["metric_diff"].values())
    }


    # Save the filtered data to a new file
    with open(output_path, "w") as f:
        json.dump(improved_data, f, indent=4)

    logging.info(
        f"Saved {len(improved_data)} improved image entries to '{output_path}'"
    )


if __name__ == "__main__":
    # eval_all()
    # eval_single()
    find_improved_noise_images()
    logging.info("Evaluation completed.")
