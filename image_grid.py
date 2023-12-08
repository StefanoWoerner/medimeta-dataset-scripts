import os

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from config import config as cfg


def main():
    mode = cfg.get("mode", "micro")
    num_rows = cfg.get("num_rows", 4)
    num_cols = cfg.get("num_cols", 4)
    size_multiplier = cfg.get("size_multiplier", 1)
    dataset_ids = [
        d
        for d in os.listdir(cfg.unified_data_base_path)
        if os.path.isdir(os.path.join(cfg.unified_data_base_path, d))
    ]
    num_images = num_rows * num_cols

    if mode == "micro":
        for id in dataset_ids:
            # load random images
            image_path = os.path.join(cfg.unified_data_base_path, id, "images")
            image_names = np.random.choice(os.listdir(image_path), num_images)
            images = [
                Image.open(os.path.join(image_path, image_name)).convert("RGB")
                for image_name in image_names
            ]
            save_path = os.path.join(cfg.unified_data_base_path, id, "teaser.png")
            save_grid(images, num_cols, num_rows, size_multiplier, save_path)
    elif mode == "macro":
        # load random images
        datasets = np.random.choice(dataset_ids, num_images)
        images = []
        for dataset in datasets:
            image_path = os.path.join(cfg.unified_data_base_path, dataset, "images")
            image_name = np.random.choice(os.listdir(image_path))
            images.append(Image.open(os.path.join(image_path, image_name)).convert("RGB"))
        save_path = os.path.join(cfg.unified_data_base_path, "teaser.png")
        save_grid(images, num_cols, num_rows, size_multiplier, save_path)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    print("Done.")


def save_grid(images, num_cols, num_rows, size_multiplier, save_path):
    # create a grid of images
    fig, axes = plt.subplots(
        num_rows, num_cols, figsize=(num_cols * size_multiplier, num_rows * size_multiplier)
    )
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i])
        ax.axis("off")
    plt.tight_layout(pad=0.5)
    plt.savefig(save_path)


if __name__ == "__main__":
    main()
