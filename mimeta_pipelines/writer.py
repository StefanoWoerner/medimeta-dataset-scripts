"""Save datasets in a unified format.
"""
import os
from multiprocessing import Lock
from multiprocessing.pool import ThreadPool
from shutil import copyfile, rmtree
import tqdm

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
        dtype=np.uint8,
    ):
        """Initialize the dataset writer.
        :param out_path: path to the output directory.
        :param info_path: path to the info file.
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

        # Initialize dataset
        self.dataset_length = sum(list(self.info_dict["original_splits_num_samples"].values()))
        # original splits
        self.original_splits = [None] * self.dataset_length
        # annotations, paths
        self.add_annots = [None] * self.dataset_length
        self.old_paths = [None] * self.dataset_length
        self.new_paths = [None] * self.dataset_length
        # task labels
        self.task_labels_dict = {task["task_name"]: task["labels"] for task in self.info_dict["tasks"]}
        self.task_labels = {task_name: [None] * self.dataset_length for task_name in self.task_labels_dict.keys()}
        # base dir of the images
        self.images_relpath = "images"
        os.makedirs(os.path.join(self.out_path, self.images_relpath))
        # hdf5 file
        self.out_img_shape = self.info_dict["input_size"]
        self.hdf5_path = os.path.join(self.out_path, "images.hdf5")
        self.hdf5_dataset_name = "images"
        self.dataset_file = h5py.File(self.hdf5_path, "w")
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
        # lock for index determination
        self.index_lock = Lock()

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
            assert None not in self.new_paths
            # original splits
            original_splits_path = os.path.join(self.out_path, "original_splits")
            os.makedirs(original_splits_path)
            for split in SPLITS:
                split_paths = [
                    path for path, orig_split in zip(self.new_paths, self.original_splits) if orig_split == split
                ]
                # check coherent with info file
                assert self.info_dict["original_splits_num_samples"][split] == len(split_paths)
                if split_paths:
                    with open(os.path.join(original_splits_path, f"{split}.txt"), "w") as f:
                        f.write("\n".join(split_paths))

            # task labels (1 file per task)
            task_labels_path = os.path.join(self.out_path, "task_labels")
            os.makedirs(task_labels_path)
            for task_name, task_list in self.task_labels.items():
                npy_save_path = os.path.join(task_labels_path, task_name)
                np.save(npy_save_path, task_list)

            # annotations
            annotations_path = os.path.join(self.out_path, "annotations.csv")
            annotations_df = pd.DataFrame.from_records(self.add_annots, index=self.new_paths)
            annotations_df.index.name = "filepath"
            annotations_df["original_split"] = self.original_splits
            annotations_df["original_filepath"] = self.old_paths
            # tasks
            task_col_names = []
            for task_name, task_dict in self.task_labels_dict.items():
                task_col_name = f"tasks/{task_name}"
                assert task_col_name not in annotations_df.columns
                task_col_names.append(task_col_name)
                annotations_df[task_col_name] = self.task_labels[task_name]
                assert task_name not in annotations_df.columns
                task_def = [task for task in self.info_dict["tasks"] if task["task_name"] == task_name][0]
                if task_def["task_target"] == "MULTILABEL_CLASSIFICATION":
                    annotations_df[task_name] = annotations_df[task_col_name].apply(
                        lambda labels: "|".join([task_dict[i] for i, label in enumerate(labels) if label == 1])
                    )
                else:
                    annotations_df[task_name] = annotations_df[task_col_name].map(task_dict)
            # reorder columns
            remaining_cols = set(annotations_df.columns)
            ordered_cols = []
            # first, tasks, original path, and split
            first_cols = task_col_names + ["original_filepath", "original_split"]
            ordered_cols += first_cols
            remaining_cols -= set(first_cols)
            # second, patient id (if available)
            if "patient_id" in remaining_cols:
                ordered_cols.append("patient_id")
                remaining_cols -= set(["patient_id"])
            # third, the rest ordered alphabetically
            ordered_cols += sorted(remaining_cols)
            annotations_df = annotations_df[ordered_cols]
            # save
            annotations_df.to_csv(annotations_path)
            # test well-formed
            annot_df = pd.read_csv(annotations_path)
            assert annot_df.columns[0] == "filepath"
            assert len(annot_df) == len(os.listdir(os.path.join(self.out_path, self.images_relpath)))

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

    def save_image_with_index(
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
        return self.save_image_with_index(image, index, self.images_relpath, check_dim=True, check_channels=True)

    def write(
        self,
        old_path: str,
        original_split: str,
        task_labels: dict[str, int],
        image: Image.Image,
        add_annots: dict | None = None,
        index: int | None = None,
    ):
        """Add labels, additional information, and image.
        :param old_path: path to the original image (relative)
        :param original_split: original split (train, val, test)
        :param task_labels: dict of task labels
        :param add_annots: dict of additional annotations
        :param image: PIL image
        :param index: index of the image in the dataset
        """
        # Basic checks
        assert original_split in SPLITS, f"Split must be one of {SPLITS}, not {original_split}."
        in_tasks = set(task_labels.keys())
        assert in_tasks == set(
            self.task_labels.keys()
        ), f"Must provide tasks {list(self.task_names.keys())}, got {in_tasks}."
        for task_name, task_label in task_labels.items():
            if isinstance(task_label, list):  # multilabel classification
                assert len(task_label) == len(self.task_labels_dict[task_name]), (
                    f"Label {task_label} invalid for task {task_name}: "
                    f"should be a list of {len(self.task_labels_dict[task_name])} elements."
                )
                assert not set(task_label) - set(
                    [0, 1]
                ), f"Label {task_label} invalid for task {task_name}: should be binary list."
            else:  # single label
                assert (
                    task_label in self.task_labels_dict[task_name]
                ), f"Label {task_label} invalid for task {task_name}."
        assert index is None or (
            index >= 0 and index < self.dataset_length and self.new_paths[index] is None
        ), f"Invalid index {index}: should be in [0, {self.dataset_length}) and not already occupied."
        # Image index
        if index is None:
            self.index_lock.acquire()  # thread-safe index determination
            index = self.old_paths.index(None)
            self.old_paths[index] = old_path
            self.index_lock.release()
        else:
            self.old_paths[index] = old_path
        # Additional annotations (if any)
        if add_annots is None:
            add_annots = dict()
        # Images
        filepath = self.save_dataset_image(image, index)
        # Register new information
        self.original_splits[index] = original_split
        for task_name in self.task_labels:
            self.task_labels[task_name][index] = task_labels[task_name]
        self.add_annots[index] = add_annots
        self.new_paths[index] = filepath

    def write_many(
        self,
        old_paths: list[str],
        original_splits: list[str],
        task_labels: list[dict[str, int]],
        images: list[Image.Image],
        add_annots: list[dict] | None = None,
        indeces: list[int] | None = None,
    ):
        """Add labels, additional information, and images.
        :param old_paths: list of paths to the original images (relative)
        :param original_splits: list of original splits (train, val, test)
        :param task_labels: list of task labels (1 dict per sample)
        :param add_annots: list of additional annotations (1 dict per sample)
        :param images: list of PIL images
        :param indeces: list of the indeces of the images in the dataset
        """
        if add_annots is None:
            add_annots = [None] * len(old_paths)
        if indeces is None:
            indeces = [None] * len(old_paths)
        with ThreadPool() as pool:
            pool.starmap(self.write, zip(old_paths, original_splits, task_labels, images, add_annots, indeces))

    def write_from_dataframe(self, df: pd.DataFrame, processing_func: callable):
        """Write whole dataset from a correctly formatted dataframe.
        :param df: dataframe containing task columns, original_filepath, original_split and additional annotations (the rest).
        :param processing_func: function that takes a row of the dataframe and returns a PIL image and a dictionary of additional annotations.
        """
        # check input dataframe
        required_cols = ["original_filepath", "original_split"] + list(self.task_labels.keys())
        assert len(set(required_cols) - set(df.columns)) == 0, f"Columns {required_cols} are required."
        # if the dataframe is not correctly indexed, reindex assuming the passed index is in the desired order
        if not df.index.min() == 0 and df.index.max() == len(df) - 1:
            df.sort_index(inplace=True)
            df.reset_index(inplace=True, drop=True)
        # dataframe to list of dicts: one per row, index included by resetting index
        df_dict = df.reset_index().transpose().to_dict().values()

        add_annot_df_cols = set(df.columns) - set(required_cols)

        def _process_image(row):
            image, add_annots = processing_func(row)
            self.write(
                old_path=row["original_filepath"],
                original_split=row["original_split"],
                task_labels={task: row[task] for task in self.task_labels.keys()},
                image=image,
                add_annots={
                    **add_annots,  # from image processing
                    **{c: row[c] for c in add_annot_df_cols},  # from dataframe
                },
                index=row["index"],
            )

        with ThreadPool() as pool:
            for _ in tqdm.tqdm(
                pool.imap_unordered(_process_image, df_dict),
                total=len(df_dict),
                desc=f"Processing {self.info_dict['name']}",
            ):
                pass
