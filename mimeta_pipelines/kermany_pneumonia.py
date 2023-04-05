"""Saves the Kermany pneumonia dataset in the unified format.

INPUT DATA:
Expects zip file as downloaded from https://data.mendeley.com/datasets/rscbjbr9sj/2
at ORIGINAL_DATA_PATH/Kermany_Pneumonia/ChestXRay2017.zip if zipped=True,
or extracted folder in ORIGINAL_DATA_PATH/Kermany_Pneumonia if zipped=False.

DATA MODIFICATIONS:
- The 283 images in RGB format are converted to grayscale using the PIL.Image.convert method.
- The images are center-cropped with the smallest dimension to obtain a square image.
- The images are resized to 224x224 (some upsized, since smaller than 224x224)
  using the PIL.Image.resize method with BICUBIC interpolation.
"""

import os
import yaml
from PIL import Image
from multiprocessing.pool import ThreadPool
from shutil import rmtree
from tqdm import tqdm
from zipfile import ZipFile
from .image_utils import center_crop
from .paths import INFO_PATH, ORIGINAL_DATA_PATH, UNIFIED_DATA_PATH, folder_paths
from .writer import UnifiedDatasetWriter


def get_unified_data(
    in_path=os.path.join(ORIGINAL_DATA_PATH, "Kermany_Pneumonia"),
    out_path=os.path.join(UNIFIED_DATA_PATH, "kermany_pneumonia"),
    info_path=os.path.join(INFO_PATH, "Kermany_Pneumonia.yaml"),
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
        with ZipFile(os.path.join(root_path, "ChestXRay2017.zip"), "r") as tf:
            tf.extractall(in_path)
        root_path = in_path
    in_path = os.path.join(in_path, "chest_xray")

    def get_img_annotation_isrgb_triple(path: str):
        img = Image.open(path)
        # some images are RGB
        isrgb = False
        if img.mode == "RGB":
            isrgb = True
            img = img.convert("L")
        # center-crop
        w, h = img.size
        img = center_crop(img)
        # resize
        img = img.resize(out_img_size, resample=Image.Resampling.BICUBIC)
        # add annotation
        add_annot = [(w, h), w / h]
        return img, add_annot, isrgb

    with UnifiedDatasetWriter(
        out_path, info_path, add_annot_cols=["original_size", "original_ratio", "disease_label"]
    ) as writer:
        for split, split_root_path in (
            ("train", os.path.join(in_path, "train")),
            ("test", os.path.join(in_path, "test")),
        ):
            class_to_idx = {v: k for k, v in info_dict["tasks"][0]["labels"].items()}
            batches = folder_paths(root=split_root_path, batch_size=batch_size, dir_to_cl_idx=class_to_idx)
            rgb_counter = 0
            for paths, labs in tqdm(batches, desc=f"Processing Kermany_Pneumonia ({split} split)"):
                with ThreadPool() as pool:
                    results = pool.map(get_img_annotation_isrgb_triple, paths)
                writer.write(
                    old_paths=[os.path.relpath(p, root_path) for p in paths],
                    original_splits=[split] * len(paths),
                    task_labels=[[lab] for lab in labs],
                    images=[res[0] for res in results],
                    add_annots=[res[1] + [info_dict["tasks"][0]["labels"][lab]] for res, lab in zip(results, labs)],
                )
                rgb_counter += sum([res[2] for res in results])
            print("Found {} RGB images, converted them.".format(rgb_counter))

    # delete temporary folder
    if zipped:
        rmtree(temp_path)


if __name__ == "__main__":
    get_unified_data()
