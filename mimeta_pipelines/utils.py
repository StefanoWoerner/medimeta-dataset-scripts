"""Utilities to save datasets in a unified format.
"""
import os
import h5py
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from PIL import Image
from multiprocessing.pool import ThreadPool
from shutil import copyfile, rmtree


# Base paths
INFO_PATH = os.path.join(os.path.dirname(__file__), "..", "dataset_info")
env_file_path = os.path.join(os.path.dirname(__file__), "..", ".env")
if os.path.exists(env_file_path):
    load_dotenv(dotenv_path=env_file_path)
    ORIGINAL_DATA_PATH = os.getenv("ORIGINAL_DATA_PATH")
    UNIFIED_DATA_PATH = os.getenv("UNIFIED_DATA_PATH")
else:
    ORIGINAL_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "original_data")
    UNIFIED_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "unified_data")


def center_crop(img: Image.Image) -> tuple[Image.Image, int, int]:
    """Center crop an image to make it square.
    :param img: PIL image.
    :returns: cropped image, original width, original height.
    """
    w, h = img.size
    if w < h:
        img = img.crop((0, (h - w) // 2, w, (h - w) // 2 + w))
    elif w > h:
        img = img.crop(((w - h) // 2, 0, (w - h) // 2 + h, h))
    assert img.size[0] == img.size[1] == min(img.size)
    return img, w, h


def folder_paths(
    root: str,
    batch_size: int,
    dir_to_cl_idx: dict[str, int],
    check_alphabetical: bool = True,
    check_cl_idxs_range: bool = True,
) -> list[tuple[list[str], list[int]]]:
    """Get batches of (paths, labels) from a folder class structure.
    :param root: root folder.
    :param batch_size: batch size.
    :param dir_to_cl_idx: dictionary mapping directories to class indices.
    :param check_alphabetical: check that the class names are in alphabetical order, and the indices range(len(classes)).
    :param check_cl_idxs_range: check that the class keys (indices) are equal to range(len(classes)).
    :returns: list of batches, each batch is a tuple of (paths, labels).
    """
    # alphabetical class order check
    if check_alphabetical:
        assert sorted(dir_to_cl_idx.items(), key=lambda x: x[1]) == sorted(dir_to_cl_idx.items(), key=lambda x: x[0])
    # class indices range check
    if check_cl_idxs_range:
        assert sorted(dir_to_cl_idx.values()) == list(range(len(dir_to_cl_idx)))
    # get paths and labels
    paths = []
    labels = []
    dirs = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
    assert set(dirs) == set(dir_to_cl_idx.keys())  # class_dict correct and complete
    for dir_ in dirs:
        new_paths = sorted(
            [
                os.path.join(root, dir_, f)
                for f in os.listdir(os.path.join(root, dir_))
                if f.split(".")[-1] in ("tif", "tiff", "png", "jpeg", "jpg") and not f.startswith(".")
            ]
        )
        paths.extend(new_paths)
        labels.extend([dir_to_cl_idx[dir_]] * len(new_paths))
    # create batches
    batches = [(paths[i : i + batch_size], labels[i : i + batch_size]) for i in range(0, len(paths), batch_size)]
    return batches


class UnifiedDatasetWriter:
    """Context manager to construct and save a dataset in the unified format.

    -- [`out_path']
        |
        |
        -- [original_splits]
        |   |
        |   -- train.txt
        |   |
        |   -- val.txt
        |   |
        |   -- test.txt
        |
        -- images
        |   |
        |   -- 000000.tiff
        |   |
        |   -- 000001.tiff
        |   ...
        |
        -- task_labels
        |   |
        |   -- [`task_name_1'].pt
        |   |
        |   -- [`task_name_2'].pt
        |   ...
        |
        -- annotations.csv
        |
        -- info.yaml
    """

    def __init__(
        self,
        out_path: str,
        info_path: str,
        add_annot_cols: list[str] | None = None,
        dtype=np.uint8,
    ):
        """Initialize the dataset writer.
        :param out_path: path to the output directory.
        :param info_path: path to the info file.
        :param add_annot_cols: list of additional columns to add to the annotations file.
        :param dtype: dtype of the images (for HDF5).
        """
        # Check output directory does not exist and create it
        self.out_path = out_path
        os.makedirs(self.out_path, exist_ok=False)
        # Copy in info file, load dict
        info_out_path = os.path.join(out_path, "info.yaml")
        copyfile(info_path, info_out_path)
        with open(info_out_path, "r") as f:
            self.info_dict = yaml.safe_load(f)

        # Initialize original splits
        self.original_splits_path = os.path.join(self.out_path, "original_splits")
        os.makedirs(self.original_splits_path)
        self.original_train = []
        self.original_val = []
        self.original_test = []
        # Initialize annotations
        self.add_annot_cols = add_annot_cols
        self.task_names = [task["task_name"] for task in self.info_dict["tasks"]]
        self.annotations = []
        self.annotations_cols = (
            ["filepath", "original_filepath", "original_split"]
            + [f"tasks/{task_name}" for task_name in self.task_names]
            + (list(self.add_annot_cols) if self.add_annot_cols else [])
        )
        # Initialize task labels
        self.task_labels = []  # shape (n_dpoints, n_tasks)
        # Create the base dir of the images
        self.images_relpath = "images"
        os.makedirs(os.path.join(self.out_path, self.images_relpath))
        # Initialize HDF5 file
        self.out_img_shape = self.info_dict["input_size"]
        self.hdf5_path = os.path.join(self.out_path, "images.hdf5")
        self.hdf5_dataset_name = "images"
        self.dataset_file = h5py.File(self.hdf5_path, "w")
        dataset_length = sum(list(self.info_dict["num_samples"].values()))
        self.dataset_file.create_dataset(
            self.hdf5_dataset_name,
            shape=(dataset_length, *self.out_img_shape[1:], self.out_img_shape[0]),  # (n_dpoints, H, W, C) size
            dtype=dtype,
        )
        # Initialize image counter
        self.current_idx = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # If error occurred, roll back
        if exc_type is not None:
            rmtree(self.out_path, ignore_errors=True)
            return
        # Close HDF5 file
        self.dataset_file.close()
        # Write out files
        try:
            # original splits
            if self.original_train:
                with open(os.path.join(self.original_splits_path, "train.txt"), "w") as f:
                    f.write("\n".join(self.original_train))
            if self.original_val:
                with open(os.path.join(self.original_splits_path, "val.txt"), "w") as f:
                    f.write("\n".join(self.original_val))
            if self.original_test:
                with open(os.path.join(self.original_splits_path, "test.txt"), "w") as f:
                    f.write("\n".join(self.original_test))
            # task labels (1 file per task)
            task_labels_path = os.path.join(self.out_path, "task_labels")
            os.makedirs(task_labels_path)
            for task_name, task_labeling in zip(self.task_names, zip(*self.task_labels)):
                npy_save_path = os.path.join(task_labels_path, task_name)
                np.save(npy_save_path, np.array(task_labeling))

            # annotations
            annotations_path = os.path.join(self.out_path, "annotations.csv")
            annotations_df = pd.DataFrame.from_records(
                data=self.annotations, columns=self.annotations_cols, index=self.annotations_cols[0]
            )
            # reorder columns
            task_names = [f"tasks/{task_name}" for task_name in self.task_names]
            ordered_annot_cols = (
                ["patient_id", *sorted(set(self.add_annot_cols) - set(["patient_id"]))]
                if "patient_id" in self.add_annot_cols
                else list(sorted(self.add_annot_cols))
            )
            ordered_cols = [*task_names, "original_filepath", "original_split", *ordered_annot_cols]
            annotations_df = annotations_df[ordered_cols]
            annotations_df.to_csv(annotations_path)
            # test well-formed
            annot_df = pd.read_csv(annotations_path)
            assert annot_df.columns[0] == "filepath"
            assert (
                len(annot_df)
                == len(os.listdir(os.path.join(self.out_path, self.images_relpath)))
                == (len(self.original_train) + len(self.original_val) + len(self.original_test))
            )
            # check coherent with info file
            assert self.info_dict["num_samples"]["train"] == len(self.original_train)
            assert self.info_dict["num_samples"]["val"] == len(self.original_val)
            assert self.info_dict["num_samples"]["test"] == len(self.original_test)

        # Roll back whenever an error occurs
        except Exception as e:
            rmtree(self.out_path, ignore_errors=True)
            raise e

    def write(
        self,
        old_paths: list[str],
        original_splits: list[str],
        task_labels: list[list[int]],
        add_annots: list | None = None,
        images: list[Image.Image] | None = None,
    ):
        """Add labels, additional information, and images.
        :param old_paths: list of paths to the original images (relative)
        :param original_splits: list of original splits (train, val, test)
        :param task_labels: list of task labels (1 list per sample)
        :param add_annots: list of additional annotations (1 list per sample)
        :param images: list of PIL images
        """
        batch_size = len(old_paths)

        # Filenames: {index_6_digits}.tiff
        filepaths = [
            os.path.join(self.images_relpath, f"{file_idx:06d}.tiff")
            for file_idx in range(self.current_idx, self.current_idx + batch_size)
        ]
        self.current_idx += batch_size

        # Additional annotations (if any)
        if add_annots is None:
            if self.add_annot_cols:
                raise TypeError("add_annot is required if add_annot_cols is not None.")
            add_annots = [[]] * batch_size

        # Images
        def save_fun(img, path):
            assert len(img.getbands()) == self.info_dict["input_size"][0]
            assert img.size == tuple(self.info_dict["input_size"][1:])
            img.save(fp=os.path.join(self.out_path, path), compression=None, quality=100)

        # in directory
        with ThreadPool() as pool:
            assert len(set(filepaths)) == len(filepaths)
            pool.starmap(save_fun, zip(images, filepaths))

        # in hdf5
        def img_to_np(img):
            arr_img = np.array(img)
            if len(img.getbands()) == 1:
                arr_img = arr_img[:, :, np.newaxis]
            return arr_img

        ds = self.dataset_file[self.hdf5_dataset_name]
        ds[self.current_idx - batch_size : self.current_idx] = [img_to_np(img) for img in images]

        # Check coherent lengths
        if not all(
            len(ls) == batch_size
            for ls in (
                filepaths,
                old_paths,
                original_splits,
                task_labels,
                add_annots,
            )
        ) or (images and len(images) != batch_size):
            raise ValueError("All arguments should have the same length.")
        # Check splits valid
        if not all(split in ("train", "val", "test") for split in original_splits):
            raise ValueError("Original splits must be of ('train', 'val', 'test').")

        # Register new information
        self.original_train += [fp for fp, split in zip(filepaths, original_splits) if split == "train"]
        self.original_val += [fp for fp, split in zip(filepaths, original_splits) if split == "val"]
        self.original_test += [fp for fp, split in zip(filepaths, original_splits) if split == "test"]
        self.task_labels += task_labels
        self.annotations += [
            [fp, orig_path, orig_split] + list(task_lab) + list(add_annot)
            for fp, orig_path, orig_split, task_lab, add_annot in zip(
                filepaths, old_paths, original_splits, task_labels, add_annots
            )
        ]
