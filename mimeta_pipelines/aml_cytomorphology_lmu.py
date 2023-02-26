"""Saves the Munich AML Cytomorphology dataset in the unified format.

INPUT DATA:
Expects annotations.dat file and AML-Cytomorphology folder as downloaded from
https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=61080958#610809587633e163895b484eafe5794e2017c585
in ORIGINAL_DATA_PATH/AML-Cytomorphology_LMU if zipped=False,
or that folder compressed at ORIGINAL_DATA_PATH/AML-Cytomorphology_LMU/AML-Cytomorphology_LMU.zip if zipped=True.

DATA MODIFICATIONS:
- The images are resized to 224x224 using the PIL.Image.thumbnail method with BICUBIC interpolation.
- The images are converted to RGB using the PIL.Image.convert method to remove the alpha channel.
"""

import numpy as np
import os
import pandas as pd
import yaml
from PIL import Image
from multiprocessing.pool import ThreadPool
from shutil import rmtree
from tqdm import tqdm
from zipfile import ZipFile
from .utils import INFO_PATH, ORIGINAL_DATA_PATH, UNIFIED_DATA_PATH, UnifiedDatasetWriter, center_crop, folder_paths


def get_unified_data(
    in_path=os.path.join(ORIGINAL_DATA_PATH, "AML-Cytomorphology_LMU"),
    out_path=os.path.join(UNIFIED_DATA_PATH, "AML-Cytomorphology_LMU"),
    info_path=os.path.join(INFO_PATH, "AML-Cytomorphology_LMU.yaml"),
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
        in_path = f"{out_path}_temp"
        with ZipFile(os.path.join(root_path, "AML-Cytomorphology_LMU.zip"), "r") as zf:
            zf.extractall(in_path)
        # change path to extracted folder
        root_path = in_path

    with UnifiedDatasetWriter(
        out_path, info_path, add_annot_cols=["annotation", "first_reannotation", "second_reannotation", "original_size"]
    ) as writer:
        images_path = os.path.join(root_path, "AML-Cytomorphology")
        class_to_idx = {v.split(" ")[0]: k for k, v in info_dict["tasks"][0]["labels"].items()}
        batches = folder_paths(root=images_path, batch_size=batch_size, class_dict=class_to_idx)
        annotations = pd.read_csv(
            os.path.join(root_path, "annotations.dat"),
            sep=r"\s+",
            names=["path", "annotation", "first_reannotation", "second_reannotation"],
            index_col=0,
        )

        def get_img_annotation_pair(path: str):
            image = Image.open(path)
            # cut alpha borders
            mask = np.array(image.getchannel("A")) == 255
            content_rows = np.argwhere(mask.any(axis=0))
            content_cols = np.argwhere(mask.any(axis=1))
            # crop black pixels out
            image = image.crop((content_rows.min(), content_cols.min(), content_rows.max() + 1, content_cols.max() + 1))
            # make squared
            image, w, h = center_crop(image)
            # remove alpha channel
            image = image.convert("RGB")
            # resize
            image.thumbnail(out_img_size, resample=Image.Resampling.BICUBIC)

            orig_size = (w, h)
            rel_path = os.path.join(*(path.split(os.sep)[-2:]))
            annot = annotations.loc[rel_path]
            # "" since NaN being a float, we would get a float column
            add_annot = [
                annot.annotation,
                annot.first_reannotation,
                annot.second_reannotation,
                orig_size,
            ]
            return image, add_annot

        for paths, labs in tqdm(batches, desc="Processing AML-Cytomorphology_LMU"):
            with ThreadPool() as pool:
                imgs_annots = pool.map(get_img_annotation_pair, paths)
            writer.write(
                old_paths=[os.path.relpath(p, root_path) for p in paths],
                original_splits=["train"] * len(paths),
                task_labels=[[lab] for lab in labs],
                images=[img_annot[0] for img_annot in imgs_annots],
                add_annots=[img_annot[1] for img_annot in imgs_annots],
            )

    # remove extracted folder to free up space
    if zipped:
        rmtree(in_path, ignore_errors=True)


if __name__ == "__main__":
    get_unified_data()
