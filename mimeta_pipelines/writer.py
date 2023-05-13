"""Save datasets in a unified format.
"""
import os
from multiprocessing.pool import ThreadPool
from shutil import copyfile, rmtree

import h5py
import numpy as np
import pandas as pd
import yaml
from PIL import Image


SPLITS = ("train", "val", "test")


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

        # Copy license file
        license_string = self.info_dict["license"]
        if "CC BY-NC-SA 4.0" in license_string:
            license_name = "CC BY-NC-SA.txt"
        elif "CC BY-SA 4.0" in license_string:
            license_name = "CC BY-SA.txt"
        else:
            raise ValueError(f"Unrecognized license: {license_string}")
        license_in_path = os.path.join(os.path.dirname(info_path), "licenses", license_name)
        license_out_path = os.path.join(out_path, "LICENSE")
        copyfile(license_in_path, license_out_path)

        # Initialize original splits
        self.original_splits = []
        # Initialize annotations
        self.add_annot_cols = add_annot_cols
        self.task_names = [task["task_name"] for task in self.info_dict["tasks"]]
        self.annotations = []
        self.task_column_names = [f"tasks/{task_name}" for task_name in self.task_names]
        self.annotations_cols = (
            ["filepath", "original_filepath", "original_split"]
            + self.task_column_names
            + (list(self.add_annot_cols) if self.add_annot_cols else [])
        )
        self.new_paths = []
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
        self.dataset_length = sum(list(self.info_dict["original_splits_num_samples"].values()))
        if self.out_img_shape[0] > 1:
            dataset_shape = (self.dataset_length, *self.out_img_shape[1:], self.out_img_shape[0])
        else:
            dataset_shape = (self.dataset_length, *self.out_img_shape[1:])
        self.dtype = dtype
        self.dataset_file.create_dataset(
            self.hdf5_dataset_name,
            shape=dataset_shape,
            dtype=self.dtype,
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
            # Original splits
            original_splits_path = os.path.join(self.out_path, "original_splits")
            os.makedirs(original_splits_path)
            for split in SPLITS:
                with open(os.path.join(original_splits_path, f"{split}.txt"), "w") as f:
                    split_paths = [
                        path for path, orig_split in zip(self.new_paths, self.original_splits) if orig_split == split
                    ]
                    # check coherent with info file
                    assert self.info_dict["original_splits_num_samples"][split] == len(split_paths)
                    f.write("\n".join(split_paths))
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
            task_names = self.task_column_names
            ordered_annot_cols = (
                ["patient_id", *sorted(set(self.add_annot_cols) - set(["patient_id"]))]
                if "patient_id" in self.add_annot_cols
                else list(sorted(self.add_annot_cols))
            )
            ordered_cols = [*self.task_column_names, "original_filepath", "original_split", *ordered_annot_cols]
            annotations_df = annotations_df[ordered_cols]
            annotations_df.to_csv(annotations_path)
            # test well-formed
            annot_df = pd.read_csv(annotations_path)
            assert annot_df.columns[0] == "filepath"
            assert (
                len(annot_df)
                == len(os.listdir(os.path.join(self.out_path, self.images_relpath)))
                == len(self.new_paths)
            )

        # Roll back whenever an error occurs
        except Exception as e:
            rmtree(self.out_path, ignore_errors=True)
            raise e

    def _image_name_from_index(self, index: int) -> str:
        return f"{index:06d}.tiff"

    def save_image(self, image: Image.Image, rel_path: str, check_dim: bool = False, check_channels: bool = False):
        """Save image to path.
        :param image: PIL image
        :param rel_path: relative path
        """
        if check_channels:
            assert len(image.getbands()) == self.out_img_shape[0]
        if check_dim:
            assert image.size == tuple(self.out_img_shape[1:])
        image.save(fp=os.path.join(self.out_path, rel_path), compression=None, quality=100)  # TODO: subprocess?

    def save_image_from_index(
        self, image: Image.Image, index: int, rel_dirpath: str, check_dim: bool = True, check_channels: bool = False
    ) -> str:
        """Save image to path.
        :param image: PIL image
        :param index: index
        :param rel_dirpath: relative directory path
        :return: relative path where the image was saved
        """
        rel_path = os.path.join(rel_dirpath, self._image_name_from_index(index))
        self.save_image(image, rel_path, check_dim, check_channels)
        return rel_path

    def save_dataset_image(self, image: Image.Image, index: int) -> str:
        """Save dataset (training) image to images directory and HDF5.
        :param image: PIL image
        :param index: index
        :return: relative path where the image was saved
        """
        # in HDF5
        ds = self.dataset_file[self.hdf5_dataset_name]
        ds[index] = np.array(image)
        # in directory
        return self.save_image(image, index, self.images_relpath, check_dim=True, check_channels=True)

    def write(
        self,
        old_path: str,
        original_split: str,
        task_labels: list[int],
        image: Image.Image,
        add_annots: list | None = None,
    ):
        """Add labels, additional information, and image.
        :param old_path: path to the original image (relative)
        :param original_split: original split (train, val, test)
        :param task_labels: list of task labels
        :param add_annots: list of additional annotations
        :param image: PIL image
        """
        # File indeces
        file_idx = self.current_idx
        self.current_idx += 1
        # Additional annotations (if any)
        if add_annots is None:
            if self.add_annot_cols:
                raise TypeError("add_annot is required if add_annot_cols is not None.")
            add_annots = []
        # Images
        filepath = self.save_dataset_image(image, file_idx)
        # Check splits valid
        if original_split not in SPLITS:
            raise ValueError(f"Original split must be of {SPLITS}.")
        # Check labels valid
        for i, task in enumerate(self.info_dict["tasks"]):
            if not task_labels[i] in task["labels"]:
                raise ValueError(f"Task {task['task_name']} label must be in {task['labels'].keys()}.")
        # Register new information
        self.original_splits.append(original_split)
        self.task_labels.append(task_labels)
        self.annotations.append([filepath, old_path, original_split] + task_labels + add_annots)
        self.new_paths.append(filepath)

    def write_many(
        self,
        old_paths: list[str],
        original_splits: list[str],
        task_labels: list[list[int]],
        images: list[Image.Image],
        add_annots: list | None = None,
    ):
        """Add labels, additional information, and images.
        :param old_paths: list of paths to the original images (relative)
        :param original_splits: list of original splits (train, val, test)
        :param task_labels: list of task labels (1 list per sample)
        :param add_annots: list of additional annotations (1 list per sample)
        :param images: list of PIL images
        """
        for old_path, original_split, task_label, image in zip(old_paths, original_splits, task_labels, images):
            self.write(old_path, original_split, task_label, image, add_annots)
