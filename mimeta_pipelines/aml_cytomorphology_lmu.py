"""Saves the NCT-CRC dataset in the unified format.

Expects zip file as downloaded from https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=61080958#610809587633e163895b484eafe5794e2017c585 (if zipped=True),
or extracted folder (if zipped=False),
in ORIGINAL_DATA_PATH/AML-Cytomorphology_LMU named AML-Cytomorphology_LMU[.zip].
"""

import os
import pandas as pd
from PIL import Image
from multiprocessing.pool import ThreadPool
from shutil import rmtree
from tqdm import tqdm
from torch.utils.data import DataLoader
from zipfile import ZipFile
from .utils import INFO_PATH, ORIGINAL_DATA_PATH, UNIFIED_DATA_PATH, UnifiedDatasetWriter, ImageFolderPaths


def get_unified_data(
    in_path=os.path.join(ORIGINAL_DATA_PATH, "AML-Cytomorphology_LMU"),
    out_path=os.path.join(UNIFIED_DATA_PATH, "AML-Cytomorphology_LMU"),
    info_path=os.path.join(INFO_PATH, "AML-Cytomorphology_LMU.yaml"),
    batch_size=256,
    out_img_size=(224, 224),
    zipped=True,
):
    root_path = os.path.join(in_path, "AML-Cytomorphology_LMU")
    # extract folder
    if zipped:
        with ZipFile(f"{root_path}.zip", 'r') as zf:
            zf.extractall(os.path.join(in_path))

    images_path = os.path.join(root_path, "AML-Cytomorphology_LMU")
    dataset = ImageFolderPaths(root=images_path, loader=lambda p: os.path.exists(p))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    annotations = pd.read_csv(
        os.path.join(root_path, "annotations.dat"), sep=r"\s+",
        names=["path", "annotation", "first_reannotation", "second_reannotation"],
        index_col=0,
    )
    # class lookup
    cls_to_idx = dataset.class_to_idx

    def pil_image(path: str):
        image = Image.open(path)
        orig_size = image.size
        rel_path = os.path.join(*(path.split(os.sep)[-2:]))
        annot = annotations.loc[rel_path]
        # "" since NaN being a float, we would get a float column
        add_annot = [
            cls_to_idx.get(annot.first_reannotation, ""),
            cls_to_idx.get(annot.second_reannotation, ""),
            orig_size
        ]
        # resize
        image.thumbnail(out_img_size, Image.ANTIALIAS)
        # remove alpha channel
        image = image.convert('RGB')
        return image, add_annot

    with UnifiedDatasetWriter(
        out_path, info_path, add_annot_cols=["first_reannotation", "second_reannotation", "original_size"]
    ) as writer:
        n_threads = 16
        for _, labs, paths in tqdm(dataloader, desc="Processing AML-Cytomorphology_LMU"):
            with ThreadPool(n_threads) as pool:
                imgs_annots = pool.map(pil_image, paths)
            writer.write(
                old_paths=list(paths),
                original_splits=["train"] * len(paths),
                task_labels=[[int(lab)] for lab in labs],
                images=[img_annot[0] for img_annot in imgs_annots],
                add_annots=[img_annot[1] for img_annot in imgs_annots],
            )

        # remove extracted folder to free up space
        if zipped:
            rmtree(root_path, ignore_errors=True)


if __name__ == "__main__":
    get_unified_data()
