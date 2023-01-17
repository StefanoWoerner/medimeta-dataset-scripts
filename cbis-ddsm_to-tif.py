import pydicom
import torch

import pandas as pd
import os
from PIL import Image

#%%

root_path = "/home/stefano/Datasets/CBIS-DDSM/"
label_file_train = os.path.join(root_path, "mass_case_description_train_set.csv")
label_file_test = os.path.join(root_path, "mass_case_description_test_set.csv")
#label_file_train = os.path.join(root_path, "calc_case_description_train_set.csv")
#label_file_test = os.path.join(root_path, "calc_case_description_test_set.csv")
df = pd.read_csv(label_file_train)
#df_test = pd.read_csv(label_file_test)
#df = pd.concat([df, df_test], axis=0)
series = "mass_train"

labels = df["pathology"]

cropped_img_paths = df["cropped image file path"]
cropped_img_paths = [
    os.path.join(root_path, "CBIS-DDSM", p.replace("\n", "")) for p in cropped_img_paths
]

mask_paths = df["ROI mask file path"]
mask_paths = [
    os.path.join(root_path, "CBIS-DDSM", p.replace("\n", "")) for p in mask_paths
]

uncropped_img_paths = df["image file path"]
uncropped_img_paths = [
    os.path.join(root_path, "CBIS-DDSM", p.replace("\n", ""))
    for p in uncropped_img_paths
]

#%%

os.makedirs(series, exist_ok=True)
for i, r in df.iterrows():
    s = sorted((os.path.getsize(p), p) for p in [uncropped_img_paths[i], mask_paths[i], cropped_img_paths[i]])
    dcm = pydicom.read_file(s[0][1])
    im = Image.fromarray(dcm.pixel_array)
    savename = f"{df['patient_id'][i]}_{df['left or right breast'][i]}_{df['image view'][i]}_{df['abnormality id'][i]}_cropped.tiff"
    im.save(os.path.join(series, savename))


#%%

for i in inds:
    for l, pref in zip(
        [uncropped_img_paths, mask_paths, cropped_img_paths],
        ["uncropped", "mask", "cropped"],
    ):
        path = l[i]
        dcm = pydicom.read_file(path)
        im = Image.fromarray(dcm.pixel_array)
        saveto = f"{df['patient_id'][i]}_{df['left or right breast'][i]}_{df['image view'][i]}_abnormality{df['abnormality id'][i]}_{pref}.png"
        im.save(saveto, "PNG")


#%%

concepts_shape, concepts_margins = concepts
single_concepts_shape = sorted(set(x for c in concepts_shape for x in c.split("-")))
single_concepts_margins = sorted(set(x for c in concepts_margins for x in c.split("-")))
