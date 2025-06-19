import os
from PIL import Image, ImageFilter
from skimage import io
import shutil

SOURCE_DIR = "remove_output"
DEST_DIR = os.path.join(SOURCE_DIR, "with_blur")

os.makedirs(DEST_DIR, exist_ok=True)


def add_gaussian_blur(image_path):
    image = io.imread(image_path)
    pil_image = Image.fromarray(image)
    return pil_image.filter(ImageFilter.GaussianBlur(radius=2))


def is_target_image(filename):
    return filename.startswith("removed_small_") or filename.startswith(
        "removed_large_"
    )


def main():
    count = 0
    for filename in os.listdir(SOURCE_DIR):
        if not is_target_image(filename):
            continue

        src_path = os.path.join(SOURCE_DIR, filename)
        dst_path = os.path.join(DEST_DIR, filename)

        try:
            blurred = add_gaussian_blur(src_path)
            blurred.save(dst_path)
            count += 1
        except Exception as e:
            print(f"⚠️ Failed to process {filename}: {e}")

    print(f"✅ Done. Applied Gaussian blur to {count} images.")
    print(f"🖼️ Blurred images saved in: {DEST_DIR}")


if __name__ == "__main__":
    main()
