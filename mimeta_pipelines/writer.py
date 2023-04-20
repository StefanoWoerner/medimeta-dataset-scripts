"""Save datasets in a unified format.
"""
import os
from multiprocessing.pool import ThreadPool
from shutil import copyfile, rmtree

import h5py
import numpy as np
import pandas as pd
import tqdm
import yaml
from PIL import Image
from multiprocessing import Lock
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

    def __init__(self, out_path: str, info_path: str, dtype=np.uint8):
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

        self.num_imgs = sum(self.info_dict["num_samples"].values())

        # Task names and labels
        self.task_labels_dict = {task["task_name"]: task["labels"] for task in self.info_dict["tasks"]}
        self.task_names = list(self.task_labels_dict.keys())
        self.task_labels = {task_name: [None] * self.num_imgs for task_name in self.task_names}

        # Initialize original splits
        self.original_splits = [None] * self.num_imgs

        # Initialize annotations and paths
        self.add_annots = [None] * self.num_imgs
        self.original_paths = [None] * self.num_imgs
        self.new_paths = [None] * self.num_imgs

        # Create the base dir of the images
        self.images_relpath = "images"
        os.makedirs(os.path.join(self.out_path, self.images_relpath))
        # Initialize HDF5 file
        self.out_img_shape = self.info_dict["input_size"]
        self.hdf5_path = os.path.join(self.out_path, "images.hdf5")
        self.hdf5_dataset_name = "images"
        self.dataset_file = h5py.File(self.hdf5_path, "w")
        dataset_length = sum(list(self.info_dict["num_samples"].values()))
        if self.out_img_shape[0] > 1:
            dataset_shape = (dataset_length, *self.out_img_shape[1:], self.out_img_shape[0])
        else:
            dataset_shape = (dataset_length, *self.out_img_shape[1:])
        self.dataset_file.create_dataset(
            self.hdf5_dataset_name,
            shape=dataset_shape,
            dtype=dtype,
        )

        # Lock for writing
        self.write_lock = Lock()

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
            self.original_splits_path = os.path.join(self.out_path, "original_splits")
            os.makedirs(self.original_splits_path)
            for split in SPLITS:
                with open(os.path.join(self.original_splits_path, f"{split}.txt"), "w") as f:
                    split_paths = [
                        path for path, orig_split in zip(self.new_paths, self.original_splits) if orig_split == split
                    ]
                    # TODO: do we always want split.txt files (in case empty), or only those present in the dataset?
                    f.write("\n".join(split_paths))

            # Task labels (1 file per task)
            task_labels_path = os.path.join(self.out_path, "task_labels")
            os.makedirs(task_labels_path)
            for task_name, task_list in zip(self.task_names, self.task_labels):
                npy_save_path = os.path.join(task_labels_path, task_name)
                np.save(npy_save_path, np.array(task_list))

            # Annotations
            annotations_path = os.path.join(self.out_path, "annotations.csv")
            annotations_df = pd.DataFrame.from_records(self.add_annots, index=self.new_paths)
            annotations_df.index.rename("filepath", inplace=True)
            annotations_df["original_split"] = self.original_splits
            annotations_df["original_filepath"] = self.original_paths
            # tasks
            task_col_names = []
            for task_name, task_dict in self.task_labels_dict.items():
                task_col_name = f"tasks/{task_name}"
                assert task_col_name not in annotations_df.columns
                task_col_names.append(task_col_name)
                annotations_df[task_col_name] = self.task_labels[task_name]
                assert task_name not in annotations_df.columns
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
            # check coherent with info file
            for split in SPLITS:
                assert self.info_dict["num_samples"][split] == (annotations_df["original_split"] == split).sum()

        # Roll back whenever an error occurs
        except Exception as e:
            rmtree(self.out_path, ignore_errors=True)
            raise e

    def image_name_from_index(self, index: int):
        return f"{index:06d}.tiff"

    def save_image(self, image: Image.Image, rel_filepath: str):
        """Save image to rel_filepath.
        :param image: PIL image
        :param rel_filepath: relative filepath to save image to
        """
        # Check image is well-formed
        expected_channels = self.out_img_shape[0]
        passed_channels = len(image.getbands())
        assert expected_channels == passed_channels, f"Expected {expected_channels} channels, got {passed_channels}."
        expected_size = tuple(self.out_img_shape[1:])
        passed_size = image.size
        assert expected_size == passed_size, f"Expected size {expected_size}, got {passed_size}."
        # Save image
        image.save(fp=os.path.join(self.out_path, rel_filepath), compression=None, quality=100)  # TODO: subprocess?

    def write(
        self,
        old_path: str,
        original_split: str,
        task_labels: dict[str, int],
        image: Image.Image,
        add_annots: dict[str, any],
        index: int | None = None,
    ):
        """Add labels, additional information, and image.
        :param old_path: path to the original image (relative)
        :param original_split: original split (train, val, or test)
        :param task_labels: dictionary of task labels
        :param image: PIL image
        :param add_annots: dictionary of additional annotations
        """
        # Basic checks
        assert (index is None) or (
            index >= 0 and index < self.num_imgs
        ), f"Index not valid, must be None or int between 0 and {self.num_imgs}"
        assert original_split in SPLITS, f"Split must be one of {SPLITS}, not {original_split}."
        in_tasks = set(task_labels.keys())
        assert in_tasks == set(self.task_names), f"Must provide tasks {self.task_names}, got {in_tasks}."
        for task_name, task_label in task_labels.items():
            assert task_label in self.task_labels_dict[task_name], f"Label {task_label} invalid for task {task_name}."

        # Lock for indexing and appending to lists
        self.write_lock.acquire()
        index = index or self.new_paths.index(None)  # if no index given, first available spot in list
        # New path with filename: {index_6_digits}.tiff
        new_path = os.path.join(self.images_relpath, self.image_name_from_index(index))
        self.new_paths[index] = new_path
        # Split
        self.original_splits[index] = original_split
        # Task labels
        for task_name in self.task_labels:
            self.task_labels[task_name][index] = task_labels[task_name]
        # Additional annotations, papths
        self.add_annots[index] = add_annots
        self.original_paths[index] = old_path

        # Save image: does not need to be locked
        self.write_lock.release()
        # (a) in directory
        self.save_image(image, new_path)
        # (b) in hdf5
        ds = self.dataset_file[self.hdf5_dataset_name]
        ds[index] = np.array(image)

    def write_many(
        self,
        old_paths: list[str],
        original_splits: list[str],
        task_labels: list[dict[str, int]],
        images: list[Image.Image],
        add_annots: list[dict[str, any]],
        indeces: list[int] | None = None,
    ):
        """Add labels, additional information, and images.
        :param old_paths: paths to the original image (relative)
        :param original_splits: original splits (train, val, or test)
        :param task_labels: list of one dictionary of task labels per image
        :param image: PIL images
        :param add_annots: list of one dictionary of additional annotations per image
        :param indeces: list of image indeces
        """
        for path, original_split, task_label, image, add_annot, index in (
            old_paths,
            original_splits,
            task_labels,
            images,
            add_annots,
            indeces or [None] * len(old_paths),
        ):
            self.write(path, original_split, task_label, image, add_annot, index)

    def write_from_dataframe(self, df: pd.DataFrame, processing_func: callable):
        """Write whole dataset from a correctly formatted dataframe.
        :param df: dataframe containing task columns, original_filepath, original_split and additional annotations (the rest).
        :param processing_func: function that takes a row of the dataframe and returns a PIL image and a dictionary of additional annotations.
        """
        # indexing must be handled by dataframe to allow multiprocessing
        assert df.index.min() == 0 and df.index.max() == len(df) - 1
        df.sort_index(inplace=True)
        assert not set(self.task_names) - set(df.columns), "Not all task columns are in the passed dataframe."
        required_cols = ["original_filepath", "original_split"]
        assert not set(required_cols) - set(df.columns), f"Columns {required_cols} are required."
        # list of dicts: one per row, index included by resetting index
        df_dict = df.reset_index().transpose().to_dict().values()

        def _process_image(row):
            image, add_annots = processing_func(row)
            self.write(
                old_path=row["original_filepath"],
                original_split=row["original_split"],
                task_labels={task: row[task] for task in self.task_names},
                image=image,
                add_annots={
                    **add_annots,
                    **{c: row[c] for c in set(df.columns) - set(required_cols) - set(self.task_names)},
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
