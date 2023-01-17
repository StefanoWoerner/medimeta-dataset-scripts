import pydicom
import matplotlib.pyplot as plt
import torch
import h5py


#%%
import pandas as pd
import os

root_path = "/home/stefano/Datasets/CBIS-DDSM/"
label_file = os.path.join(root_path, "mass_case_description_train_set.csv")
df = pd.read_csv(label_file)

labels = df["pathology"]

cropped_img_paths = df["cropped image file path"]
cropped_img_paths = [
    os.path.join(root_path, "CBIS-DDSM", p)
    for p in cropped_img_paths
]

mask_paths = df["ROI mask file path"]
mask_paths = [
    os.path.join(
        root_path, "CBIS-DDSM", p.replace("\n", "")
    )
    for p in mask_paths
]

uncropped_img_paths = df["image file path"]
uncropped_img_paths = [
    os.path.join(root_path, "CBIS-DDSM", p)
    for p in uncropped_img_paths
]

inds = df.index[df["patient_id"] == "P_00092"]

#%%
store = pd.HDFStore('test.hdf5')

store.put('dataset_01', df)

metadata = {'scale':0.1,'offset':15}

store.get_storer('dataset_01').attrs.metadata = metadata

store.close()

#%%

with h5py.File('test.hdf5', 'w') as f:
    dset = f.create_dataset("cbis_labels", data=list(labels))

#%%

with h5py.File('test.hdf5', 'r') as f:
    data = f['cbis_labels']

    # get the minimum value
    print(min(data))

    # get the maximum value
    print(max(data))

    # get the values ranging from index 0 to 15
    print(data[:15])

    d1 = data[1]

#%%

fd = h5py.File('test.hdf5', 'r')


#%%

fd.close()

#%%

concept_names = ['mass shape', 'mass margins']
concepts = [sorted(set((c for c in df[cn] if not pd.isna(c)))) for cn in concept_names]
concept_labels = [torch.tensor([[1 if x == c else 0 for c in concepts[i]] for x in df[cn]]) for i, cn in enumerate(concept_names)]
con = torch.cat(concept_labels, dim=1)

#%%
from PIL import Image

for i in inds:
    for l, pref in zip(
        [uncropped_img_paths, mask_paths, cropped_img_paths], ["uncropped", "mask", "cropped"]
    ):
        path = l[i]
        dcm = pydicom.read_file(path)
        im = Image.fromarray(dcm.pixel_array)
        saveto = f"{df['patient_id'][i]}_{df['left or right breast'][i]}_{df['image view'][i]}_abnormality{df['abnormality id'][i]}_{labels[i]}_{pref}.png"
        im.save(saveto, "PNG")
