import pandas as pd
import numpy as np
import nibabel as nib
import os
from PIL import Image

df = pd.read_csv('data.csv')
base_dir = "../../mimeta-dataset-scripts/data/original_data/LITS"
n_examples = 5
dir_path = 'bboxes_coronal'
os.makedirs(dir_path, exist_ok=True)

def draw_bounding_box(image, ys, xs):
    # Get the dimensions of the image
    height, width = image.shape
    # Ensure that the bounding box coordinates are within the image bounds
    x1, x2 = np.clip(xs, 0, [width-1, width-1])
    y1, y2 = np.clip(ys, 0, [height-1, height-1])
    # Draw the bounding box
    image[y1, x1:x2+1] = 255  # Top edge
    image[y2, x1:x2+1] = 255  # Bottom edge
    image[y1:y2+1, x1] = 255  # Left edge
    image[y1:y2+1, x2] = 255  # Right edge
    return image

def get_image(path, bbox):
    path = base_dir + "/" + path
    img = nib.load(path).get_fdata()
    img = img[:, int(0.5 * (bbox[1][0] + bbox[1][1])), :]
    img[img < -150] = -150
    img[img > 250] = 250
    img = 255 * (img + 150) / 400
    img = img.astype(np.uint8)
    return img

for label in set(df['organ_name']):
    df_sub = df[df['organ_name'] == label]
    for i in range(n_examples):
        bbox = eval(list(df_sub['bbox'])[i])
        img = get_image(list(df_sub['image_path'])[i], bbox)
        img = draw_bounding_box(img, bbox[0], bbox[2])
        Image.fromarray(img).save(f"{dir_path}/{label}_{i}.tiff")
