"""Saves the Munich AML Cytomorphology dataset in the unified format.

EXPECTED INPUT FOLDER CONTENTS:
if zipped=True (default):
- a zip file named AML-Cytomorphology_LMU.zip containing the
  AML-Cytomorphology_LMU folder described below
if zipped=False:
- a folder named AML-Cytomorphology_LMU containing:
    - the AML-Cytomorphology folder
    - the annotations.dat file
  downloaded from
  https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=61080958#610809587633e163895b484eafe5794e2017c585

DATA MODIFICATIONS:
- The images are resized to 224x224 using the PIL.Image.thumbnail method
  with BICUBIC interpolation.
- The images are converted to RGB using the PIL.Image.convert method to
  remove the alpha channel.
"""

import os
from multiprocessing.pool import ThreadPool
from shutil import rmtree
from zipfile import ZipFile

import pandas as pd
from PIL import Image
from tqdm import tqdm

from .paths import INFO_PATH, folder_paths, setup
from .writer import UnifiedDatasetWriter


def get_unified_data(
    in_path,
    info_path=os.path.join(INFO_PATH, "AML-Cytomorphology_LMU.yaml"),
    batch_size=512,
    out_img_size=(224, 224),
    zipped=True,
):
    info_dict, out_path = setup(in_path, info_path)

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
        batches = folder_paths(root=images_path, batch_size=batch_size, dir_to_cl_idx=class_to_idx)
        annotations = pd.read_csv(
            os.path.join(root_path, "annotations.dat"),
            sep=r"\s+",
            names=["path", "annotation", "first_reannotation", "second_reannotation"],
            index_col=0,
        )

        def get_img_annotation_pair(path: str):
            image = Image.open(path)
            orig_size = image.size
            rel_path = os.path.join(*(path.split(os.sep)[-2:]))
            annot = annotations.loc[rel_path]
            # "" since NaN being a float, we would get a float column
            add_annot = [
                annot.annotation,
                annot.first_reannotation,
                annot.second_reannotation,
                orig_size,
            ]
            # remove alpha channel
            image = image.convert("RGB")
            # resize
            image.thumbnail(out_img_size, resample=Image.Resampling.BICUBIC)
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


def main():
    from config import config as cfg

    pipeline_name = "aml_cyto"
    get_unified_data(**cfg.pipeline_args[pipeline_name])


if __name__ == "__main__":
    main()
