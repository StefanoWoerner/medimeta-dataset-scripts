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
import pandas as pd
from shutil import rmtree
from zipfile import ZipFile

import pandas as pd
from PIL import Image

from .paths import INFO_PATH, folder_paths, setup
from .writer import UnifiedDatasetWriter


def get_unified_data(
    in_path,
    info_path=os.path.join(INFO_PATH, "AML-Cytomorphology_LMU.yaml"),
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

    with UnifiedDatasetWriter(out_path, info_path) as writer:
        images_basedir = os.path.join(root_path, "AML-Cytomorphology")
        class_to_idx = {v.split(" ")[0]: k for k, v in info_dict["tasks"][0]["labels"].items()}
        paths, labels = folder_paths(root=images_basedir, dir_to_cl_idx=class_to_idx)
        morph_task = info_dict["tasks"][0]
        path2idx = dict(zip(paths, labels))
        annotations = pd.read_csv(
            os.path.join(root_path, "annotations.dat"),
            sep=r"\s+",
            names=["original_filepath", "annotation", "first_reannotation", "second_reannotation"],
        )
        annotations[morph_task["task_name"]] = annotations["original_filepath"].map(path2idx)
        annotations["original_split"] = "train"
        # keep only needed columns
        annotations = annotations[
            [
                "original_filepath",
                "original_split",
                morph_task["task_name"],
                "annotation",
                "first_reannotation",
                "second_reannotation",
            ]
        ]
        annotations.to_csv("annotations.csv")

        def get_img_addannot_pair(annotations_row):
            path = annotations_row["original_filepath"]
            image = Image.open(os.path.join(images_basedir, path))
            orig_size = image.size
            add_annot = {"original_image_size": orig_size}
            # remove alpha channel
            image = image.convert("RGB")
            # resize
            image.thumbnail(out_img_size, resample=Image.Resampling.BICUBIC)
            return image, add_annot

        writer.write_from_dataframe(df=annotations, processing_func=get_img_addannot_pair)

    # remove extracted folder to free up space
    if zipped:
        rmtree(in_path, ignore_errors=True)


def main():
    from config import config as cfg

    pipeline_name = "aml_cyto"
    get_unified_data(**cfg.pipeline_args[pipeline_name])


if __name__ == "__main__":
    main()
