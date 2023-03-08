"""Utilities to transform images.
"""
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
