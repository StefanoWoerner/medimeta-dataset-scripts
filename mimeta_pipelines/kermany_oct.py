"""Saves the Kermany optical coherence tomography (OCT) dataset in the unified format.

EXPECTED INPUT FOLDER CONTENTS:
if zipped=True (default):
- the OCT2017.tar.gz file downloaded from
  https://data.mendeley.com/datasets/rscbjbr9sj/2
if zipped=False:
- the extracted OCT2017 folder

DATA MODIFICATIONS:
- The images are center-cropped with the smallest dimension to obtain a
  square image.
- The images are resized to 224x224 using the PIL.Image.thumbnail method
  with BICUBIC interpolation.
"""

import os
import tarfile
from multiprocessing.pool import ThreadPool
from shutil import rmtree

from PIL import Image
from tqdm import tqdm

from .image_utils import center_crop
from .paths import INFO_PATH, folder_paths, setup
from .writer import UnifiedDatasetWriter


def get_unified_data(
    in_path,
    info_path=os.path.join(INFO_PATH, "Kermany_OCT.yaml"),
    batch_size=512,
    out_img_size=(224, 224),
    zipped=True,
):
    info_dict, out_path = setup(in_path, info_path)

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
        img = center_crop(img)
        # resize
        img.thumbnail(out_img_size, resample=Image.Resampling.BICUBIC)
        # add annotation
        add_annot = {"original_image_size": (w, h), "original_image_ratio": w / h}
        return img, add_annot

    with UnifiedDatasetWriter(out_path, info_path) as writer:
        task = info_dict["tasks"][0]
        for split, split_root_path in (
            ("train", os.path.join(in_path, "train")),
            ("test", os.path.join(in_path, "test")),
        ):
            class_to_idx = {v: k for k, v in task["labels"].items()}
            batches = folder_paths(
                root=split_root_path, dir_to_cl_idx=class_to_idx, batch_size=batch_size
            )
            for paths, labs in tqdm(batches, desc=f"Processing Kermany_OCT ({split} split)"):
                with ThreadPool() as pool:
                    imgs_annots = pool.map(get_img_annotation_pair, paths)
                writer.write_many(
                    old_paths=[os.path.relpath(p, root_path) for p in paths],
                    original_splits=[split] * len(paths),
                    task_labels=[{task["task_name"]: lab} for lab in labs],
                    images=[img_annot[0] for img_annot in imgs_annots],
                    add_annots=[img_annot[1] for img_annot in imgs_annots],
                )

    # delete temporary folder
    if zipped:
        rmtree(temp_path)


def main():
    from config import config as cfg

    pipeline_name = "kermany_oct"
    get_unified_data(**cfg.pipeline_args[pipeline_name])


if __name__ == "__main__":
    main()
