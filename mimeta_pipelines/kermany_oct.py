"""Saves the Kermany optical coherence tomography (OCT) dataset in the unified format.

INPUT DATA:
Expects tar.gz file as downloaded from https://data.mendeley.com/datasets/rscbjbr9sj/2
at ORIGINAL_DATA_PATH/Kermany_OCT/OCT2017.tar.gz if zipped=True,
or extracted folder in ORIGINAL_DATA_PATH/OCT2017 if zipped=False.

DATA MODIFICATIONS:
- The images are center-cropped with the smallest dimension to obtain a square image.
- The images are resized to 224x224 using the PIL.Image.thumbnail method with BICUBIC interpolation.
"""

import os
import tarfile
import yaml
from PIL import Image
from multiprocessing.pool import ThreadPool
from shutil import rmtree
from tqdm import tqdm
from torch.utils.data import DataLoader
from .utils import INFO_PATH, ORIGINAL_DATA_PATH, UNIFIED_DATA_PATH, UnifiedDatasetWriter, ImageFolderPaths


def get_unified_data(
    in_path=os.path.join(ORIGINAL_DATA_PATH, "Kermany_OCT"),
    out_path=os.path.join(UNIFIED_DATA_PATH, "Kermany_OCT"),
    info_path=os.path.join(INFO_PATH, "Kermany_OCT.yaml"),
    batch_size=512,
    out_img_size=(224, 224),
    zipped=True,
):
    assert not os.path.exists(out_path), f"Output path {out_path} already exists. Please delete it first."

    with open(info_path, "r") as f:
        info_dict = yaml.safe_load(f)

    root_path = in_path
    # extract folder
    if zipped:
        # extract to out_path (temporary)
        temp_path = f"{out_path}_temp"
        in_path = temp_path
        with tarfile.open(os.path.join(root_path, "OCT2017.tar.gz"), "r:gz") as tf:
            tf.extractall(in_path)
        root_path = in_path
    in_path = os.path.join(in_path, "OCT2017")

    def get_img_annotation_pair(path: str):
        img = Image.open(path)
        # center-crop
        w, h = img.size
        if w < h:
            img = img.crop((0, (h - w) // 2, w, (h - w) // 2 + w))
        elif w > h:
            img = img.crop(((w - h) // 2, 0, (w - h) // 2 + h, h))
        assert img.size[0] == img.size[1] == min(img.size)
        # resize
        img.thumbnail(out_img_size, Image.BICUBIC)
        # add annotation
        add_annot = [(w, h), w / h]
        return img, add_annot

    with UnifiedDatasetWriter(
        out_path, info_path, add_annot_cols=["original_size", "original_ratio", "disease_label"]
    ) as writer:
        for split, split_root_path in (
            ("train", os.path.join(in_path, "train")),
            ("test", os.path.join(in_path, "test")),
        ):
            # dummy loader to avoid actually loading the images
            dataset = ImageFolderPaths(root=split_root_path, loader=lambda p: os.path.exists(p))
            assert dataset.class_to_idx == {v: k for k, v in info_dict["tasks"][0]["labels"].items()}
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            for _, labs, paths in tqdm(dataloader, desc=f"Processing Kermany_OCT ({split} split)"):
                with ThreadPool() as pool:
                    imgs_annots = pool.map(get_img_annotation_pair, paths)
                writer.write(
                    old_paths=[os.path.relpath(p, root_path) for p in paths],
                    original_splits=[split] * len(paths),
                    task_labels=[[lab.item()] for lab in labs],
                    images=[img_annot[0] for img_annot in imgs_annots],
                    add_annots=[
                        img_annot[1] + [info_dict["tasks"][0]["labels"][lab.item()]]
                        for img_annot, lab in zip(imgs_annots, labs)
                    ],
                )

    # delete temporary folder
    if zipped:
        rmtree(temp_path)


if __name__ == "__main__":
    get_unified_data()
