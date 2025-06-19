import os
import torch
from PIL import Image
from tqdm import tqdm
from caption import generate_beam
from datetime import datetime
import json
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import logging
import tempfile

OUTPUT_PATH = "outputs"


def generate_captions(
    model,
    tokenizer,
    clip_model,
    preprocess,
    DATA_EVAL_PATH,
    DEVICE,
    prefix_length=77,
    folder_name="base_images",
    use_beam_search=True,
):

    eval_folder = os.path.join(DATA_EVAL_PATH, folder_name)
    image_files = [f for f in os.listdir(eval_folder) if f.lower().endswith(".jpg")]
    images_path = [os.path.join(eval_folder, f) for f in image_files]

    results = []
    image_ids = []

    for idx, img_path in tqdm(enumerate(images_path), total=len(images_path)):
        logging.info(f"Processing image {idx + 1}/{len(images_path)}: {img_path}")
        pil_image = Image.open(img_path).convert("RGB")
        image_tensor = preprocess(pil_image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            prefix = clip_model.encode_image(image_tensor).to(
                DEVICE, dtype=torch.float32
            )
            prefix = prefix / prefix.norm(2, -1).item()
            prefix_embed = model.clip_project(prefix).reshape(1, prefix_length, -1)

        # if use_beam_search:
        caption = generate_beam(model, tokenizer, embed=prefix_embed)[0]
        # else:
        #     caption = generate2(model, tokenizer, embed=prefix_embed)

        file_name = os.path.basename(img_path)
        image_id = int(file_name.split("_")[-1].split(".")[0])
        results.append({"image_id": image_id, "caption": caption})
        image_ids.append(image_id)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(OUTPUT_PATH, f"{folder_name}_{timestamp}_captions.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    return results, image_ids, output_file


def evaluate_captions(data_eval_path, resFile, image_ids, type):
    coco = COCO(data_eval_path)
    cocoRes = coco.loadRes(resFile)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params["image_id"] = image_ids
    cocoEval.evaluate()

    output_eval_file = os.path.join(OUTPUT_PATH, f"{type}_metrics.txt")
    logging.info(f"Evaluation in progress, results saved to {output_eval_file}")
    with open(output_eval_file, "w") as f:
        for metric, score in cocoEval.eval.items():
            f.write(f"{metric}: {score:.4f}\n")

    return cocoEval.eval


def evaluate_single_caption(coco, image_id, caption):
    # Create a temporary result file with one caption
    result = [{"image_id": image_id, "caption": caption}]
    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".json") as tmp:
        json.dump(result, tmp)
        tmp.flush()
        cocoRes = coco.loadRes(tmp.name)
        cocoEval = COCOEvalCap(coco, cocoRes)
        cocoEval.params["image_id"] = [image_id]
        cocoEval.evaluate()
        os.unlink(tmp.name)
        return cocoEval.eval
