"""Utilities to transform images.
"""
import numpy as np
from enum import Enum
from PIL import Image


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


Slice = Enum("Slice", ["SAGITTAL", "CORONAL", "AXIAL"])


def slice_3d_image(
    img: np.ndarray, bbox: tuple[tuple[int, int], tuple[int, int], tuple[int, int]], axis: Slice
) -> tuple[np.ndarray, tuple[tuple[int, int], tuple[int, int]]]:
    """Slice a 3D nii image and corresponing bounding box into a 2D image with the corresponding bounding box,
    for the given axis (sagittal, coronal, axial).
    :param img: 3d image in numpy array.
    :param bbox: bounding box of the 3D image in the format ((x_min, x_max), (y_min, y_max), (z_min, z_max)).
    :param axis: axis to slice the image.
    :returns: tuple of 3 tuples (sagittal, coronal, axial), each containing a 2D image and its corresponding bounding box.
    """
    assert all([img.shape[i] >= bbox[i][1] and bbox[i][1] >= bbox[i][0] for i in range(3)])
    if axis == Slice.SAGITTAL:
        img = img[(bbox[0][0] + bbox[0][1]) // 2, :, :]
        bbox = (bbox[1], bbox[2])
    elif axis == Slice.CORONAL:
        img = img[:, (bbox[1][0] + bbox[1][1]) // 2, :]
        bbox = (bbox[0], bbox[2])
    elif axis == Slice.AXIAL:
        img = img[:, :, (bbox[2][0] + bbox[2][1]) // 2]
        bbox = (bbox[0], bbox[1])
    return img, bbox


def ct_windowing(img: np.ndarray, window_width: float = 400, window_level: float = 50) -> np.ndarray:
    # default values from https://radiopaedia.org/articles/windowing-ct (abdominal window)
    """Apply windowing to a CT image.
    :param img: 2D numpy array.
    :param window_width: window width.
    :param window_level: window level.
    :returns: windowed image (float array in [0, 1])
    """
    lower = window_level - window_width / 2
    upper = window_level + window_width / 2
    img[img < lower] = lower
    img[img > upper] = upper
    img = (img - lower) / (upper - lower)
    return img


def ratio_cut(img: np.ndarray, bbox: tuple[tuple[int, int], tuple[int, int]], ratio: float = 1.0) -> np.ndarray:
    """Cut a crop with ratio width/height of an image based on a bounding box (endpoint inclusive).
    :param img: 2D numpy array.
    :param bbox: bounding box of the image in the format ((x_min, x_max), (y_min, y_max)); max values are inclusive.
    :param ratio: ratio width/height of the crop.
    :returns: cropped image.
    """
    img_shape = np.array(img.shape)
    # compute out bbox indices
    out_bbox = [list(bbox[i]) for i in range(2)]
    bbox_centers = ((bbox[0][0] + bbox[0][1]) // 2, (bbox[1][0] + bbox[1][1]) // 2)
    bbox_diff = (bbox[0][1] - bbox[0][0], bbox[1][1] - bbox[1][0])
    if bbox_diff[0] * ratio >= bbox_diff[1]:
        out_bbox[1][0] = round(bbox_centers[1] - bbox_diff[0] * ratio / 2)
        out_bbox[1][1] = out_bbox[1][0] + round(bbox_diff[0] * ratio)
    else:
        out_bbox[0][0] = round(bbox_centers[0] - bbox_diff[1] / ratio / 2)
        out_bbox[0][1] = out_bbox[0][0] + round(bbox_diff[1] / ratio)
    # initialize crop
    crop_shape = np.array([out_bbox[0][1] - out_bbox[0][0] + 1, out_bbox[1][1] - out_bbox[1][0] + 1])
    crop = np.zeros(crop_shape, dtype=img.dtype)
    # calculate img slice positions
    start = np.clip(np.array([out_bbox[0][0], out_bbox[1][0]]), a_min=0, a_max=img_shape - 1)  # where to start in img
    end = np.clip(np.array([out_bbox[0][1], out_bbox[1][1]]), a_min=0, a_max=img_shape - 1)  # where to end in img
    # calculate crop slice positions
    crop_low = start - np.array([out_bbox[0][0], out_bbox[1][0]])  # where to start in crop
    crop_high = crop_low + (end - start)  # where to end in crop
    crop[crop_low[0] : crop_high[0] + 1, crop_low[1] : crop_high[1] + 1] = img[
        start[0] : end[0] + 1, start[1] : end[1] + 1
    ]
    return crop


def test_ratio_cut():
    img = np.array([[i] * 10 for i in range(10)])
    # inside
    crop1 = ((1, 2), (1, 2))
    assert (ratio_cut(img, crop1, (2, 2)) == np.array([[1, 1], [2, 2]])).all()
    # non-square
    crop2 = ((1, 1), (2, 3))
    assert (ratio_cut(img, crop2, (2, 2)) == np.array([[1, 1], [2, 2]])).all()
    # over left border
    crop3 = ((0, 0), (0, 2))
    assert (ratio_cut(img, crop3, (3, 3)) == np.array([[0, 0, 0], [0, 0, 0], [1, 1, 1]])).all()
    # over right border
    crop4 = ((8, 9), (9, 9))
    assert (ratio_cut(img, crop4, (2, 2)) == np.array([[8, 0], [9, 0]])).all()


def draw_colored_bounding_box(
    img: np.ndarray, bbox: tuple[tuple[int, int], tuple[int, int]], color: np.ndarray | int = 255
) -> np.ndarray:
    """Draw a bounding box on an image.
    :param img: 3D numpy array (color channel last).
    :param bbox: bounding box of the image in the format ((x_min, x_max), (y_min, y_max)); max values are inclusive.
    :param color: 1D numpy array (length 3) color of the bounding box.
    :returns: image with bounding box.
    """
    img = img.copy()
    img[bbox[0][0] : bbox[0][1] + 1, bbox[1][0], :] = color
    img[bbox[0][0] : bbox[0][1] + 1, bbox[1][1], :] = color
    img[bbox[0][0], bbox[1][0] : bbox[1][1] + 1, :] = color
    img[bbox[0][1], bbox[1][0] : bbox[1][1] + 1, :] = color
    return img
