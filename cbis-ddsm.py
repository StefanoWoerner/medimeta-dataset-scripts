import pydicom
import matplotlib.pyplot as plt
import torch

#%%
path = "/home/stefano/Datasets/CBIS-DDSM/CBIS-DDSM/Mass-Training_P_00001_LEFT_CC_1/1.3.6.1.4.1.9590.100.1.2.108268213011361124203859148071588939106/1.3.6.1.4.1.9590.100.1.2.296736403313792599626368780122205399650/1-2.dcm"
path = "/home/stefano/Datasets/CBIS-DDSM/CBIS-DDSM/Mass-Training_P_00001_LEFT_CC_1/1.3.6.1.4.1.9590.100.1.2.108268213011361124203859148071588939106/1.3.6.1.4.1.9590.100.1.2.296736403313792599626368780122205399650/1-1.dcm"
path = "/home/stefano/Datasets/CBIS-DDSM/CBIS-DDSM/Mass-Training_P_00001_LEFT_CC/1.3.6.1.4.1.9590.100.1.2.422112722213189649807611434612228974994/1.3.6.1.4.1.9590.100.1.2.342386194811267636608694132590482924515/1-1.dcm"
ds = pydicom.read_file(path)

#%%
plt.figure()
plt.imshow(ds.pixel_array)
plt.show()

#%%
import pandas as pd
import os

root_path = "/home/stefano/Datasets/CBIS-DDSM/"
label_file_train = os.path.join(root_path, "mass_case_description_train_set.csv")
label_file_test = os.path.join(root_path, "mass_case_description_test_set.csv")
label_file_train = os.path.join(root_path, "calc_case_description_train_set.csv")
label_file_test = os.path.join(root_path, "calc_case_description_test_set.csv")
df = pd.read_csv(label_file_train)
df_test = pd.read_csv(label_file_test)
df = pd.concat([df, df_test], axis=0)

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

inds = df.index[df["patient_id"] == "P_00005"]

#%%

concept_names = ["mass shape", "mass margins"]
concepts = [sorted(set((c for c in df[cn] if not pd.isna(c)))) for cn in concept_names]
concept_labels = [
    torch.tensor([[1 if x == c else 0 for c in concepts[i]] for x in df[cn]])
    for i, cn in enumerate(concept_names)
]
con = torch.cat(concept_labels, dim=1)

#%%

concept_names = ["calc type", "calc distribution"]
concepts = [sorted(set((c for c in df[cn] if not pd.isna(c)))) for cn in concept_names]
concept_labels = [
    torch.tensor([[1 if x == c else 0 for c in concepts[i]] for x in df[cn]])
    for i, cn in enumerate(concept_names)
]
con = torch.cat(concept_labels, dim=1)

#%%
concept_names = ["mass shape", "mass margins"]
concepts = [
    [
        "ARCHITECTURAL_DISTORTION",
        "ASYMMETRIC_BREAST_TISSUE",
        "FOCAL_ASYMMETRIC_DENSITY",
        "IRREGULAR",
        "LOBULATED",
        "LYMPH_NODE",
        "OVAL",
        "ROUND",
    ],
    [
        "CIRCUMSCRIBED",
        "ILL_DEFINED",
        "MICROLOBULATED",
        "OBSCURED",
        "SPICULATED",
    ],
]
concept_labels = [
    torch.tensor(
        [
            [(1 if (0 if pd.isna(c) else c in x) else 0) for c in concepts[i]]
            for x in df[cn]
        ]
    )
    for i, cn in enumerate(concept_names)
]

#%%
concept_names = ["calc type", "calc distribution"]
concepts = [
    [
        "AMORPHOUS",
        "COARSE",
        "DYSTROPHIC",
        "EGGSHELL",
        "FINE_LINEAR_BRANCHING",
        "LARGE_RODLIKE",
        "LUCENT_CENTER",
        "LUCENT_CENTERED",
        "MILK_OF_CALCIUM",
        "PLEOMORPHIC",
        "PUNCTATE",
        "ROUND_AND_REGULAR",
        "SKIN",
        "VASCULAR",
    ],
    ["CLUSTERED", "DIFFUSELY_SCATTERED", "LINEAR", "REGIONAL", "SEGMENTAL"],
]

concept_labels = [
    torch.tensor(
        [
            [(1 if (0 if pd.isna(c) else c in x) else 0) for c in concepts[i]]
            for x in df[cn]
        ]
    )
    for i, cn in enumerate(concept_names)
]

#%%
from PIL import Image

for i in inds:
    for l, pref in zip(
        [uncropped_img_paths, mask_paths, cropped_img_paths],
        ["uncropped", "mask", "cropped"],
    ):
        path = l[i]
        dcm = pydicom.read_file(path)
        im = Image.fromarray(dcm.pixel_array)
        saveto = f"{df['patient_id'][i]}_{df['left or right breast'][i]}_{df['image view'][i]}_abnormality{df['abnormality id'][i]}_{labels[i]}_{pref}.png"
        im.save(saveto, "PNG")


#%%

concepts_explicit = [
    [
        "ARCHITECTURAL_DISTORTION",
        "ASYMMETRIC_BREAST_TISSUE",
        "FOCAL_ASYMMETRIC_DENSITY",
        "IRREGULAR",
        "IRREGULAR-ARCHITECTURAL_DISTORTION",
        "IRREGULAR-ASYMMETRIC_BREAST_TISSUE",
        "IRREGULAR-FOCAL_ASYMMETRIC_DENSITY",
        "LOBULATED",
        "LOBULATED-ARCHITECTURAL_DISTORTION",
        "LOBULATED-IRREGULAR",
        "LOBULATED-LYMPH_NODE",
        "LOBULATED-OVAL",
        "LYMPH_NODE",
        "OVAL",
        "OVAL-LOBULATED",
        "OVAL-LYMPH_NODE",
        "ROUND",
        "ROUND-IRREGULAR-ARCHITECTURAL_DISTORTION",
        "ROUND-LOBULATED",
        "ROUND-OVAL",
    ],
    [
        "CIRCUMSCRIBED",
        "CIRCUMSCRIBED-ILL_DEFINED",
        "CIRCUMSCRIBED-MICROLOBULATED",
        "CIRCUMSCRIBED-MICROLOBULATED-ILL_DEFINED",
        "CIRCUMSCRIBED-OBSCURED",
        "CIRCUMSCRIBED-OBSCURED-ILL_DEFINED",
        "CIRCUMSCRIBED-SPICULATED",
        "ILL_DEFINED",
        "ILL_DEFINED-SPICULATED",
        "MICROLOBULATED",
        "MICROLOBULATED-ILL_DEFINED",
        "MICROLOBULATED-ILL_DEFINED-SPICULATED",
        "MICROLOBULATED-SPICULATED",
        "OBSCURED",
        "OBSCURED-CIRCUMSCRIBED",
        "OBSCURED-ILL_DEFINED",
        "OBSCURED-ILL_DEFINED-SPICULATED",
        "OBSCURED-SPICULATED",
        "SPICULATED",
    ],
]

#%%

concepts_explicit = [
    [
        "AMORPHOUS",
        "AMORPHOUS-PLEOMORPHIC",
        "AMORPHOUS-ROUND_AND_REGULAR",
        "COARSE",
        "COARSE-LUCENT_CENTER",
        "COARSE-PLEOMORPHIC",
        "COARSE-ROUND_AND_REGULAR",
        "COARSE-ROUND_AND_REGULAR-LUCENT_CENTER",
        "COARSE-ROUND_AND_REGULAR-LUCENT_CENTERED",
        "DYSTROPHIC",
        "EGGSHELL",
        "FINE_LINEAR_BRANCHING",
        "LARGE_RODLIKE",
        "LARGE_RODLIKE-ROUND_AND_REGULAR",
        "LUCENT_CENTER",
        "LUCENT_CENTER-PUNCTATE",
        "LUCENT_CENTERED",
        "MILK_OF_CALCIUM",
        "PLEOMORPHIC",
        "PLEOMORPHIC-AMORPHOUS",
        "PLEOMORPHIC-FINE_LINEAR_BRANCHING",
        "PLEOMORPHIC-PLEOMORPHIC",
        "PUNCTATE",
        "PUNCTATE-AMORPHOUS",
        "PUNCTATE-AMORPHOUS-PLEOMORPHIC",
        "PUNCTATE-FINE_LINEAR_BRANCHING",
        "PUNCTATE-LUCENT_CENTER",
        "PUNCTATE-PLEOMORPHIC",
        "PUNCTATE-ROUND_AND_REGULAR",
        "ROUND_AND_REGULAR",
        "ROUND_AND_REGULAR-AMORPHOUS",
        "ROUND_AND_REGULAR-EGGSHELL",
        "ROUND_AND_REGULAR-LUCENT_CENTER",
        "ROUND_AND_REGULAR-LUCENT_CENTER-DYSTROPHIC",
        "ROUND_AND_REGULAR-LUCENT_CENTER-PUNCTATE",
        "ROUND_AND_REGULAR-LUCENT_CENTERED",
        "ROUND_AND_REGULAR-PLEOMORPHIC",
        "ROUND_AND_REGULAR-PUNCTATE",
        "ROUND_AND_REGULAR-PUNCTATE-AMORPHOUS",
        "SKIN",
        "SKIN-COARSE-ROUND_AND_REGULAR",
        "SKIN-PUNCTATE",
        "SKIN-PUNCTATE-ROUND_AND_REGULAR",
        "VASCULAR",
        "VASCULAR-COARSE",
        "VASCULAR-COARSE-LUCENT_CENTER-ROUND_AND_REGULAR-PUNCTATE",
        "VASCULAR-COARSE-LUCENT_CENTERED",
    ],
    [
        "CLUSTERED",
        "CLUSTERED-LINEAR",
        "CLUSTERED-SEGMENTAL",
        "DIFFUSELY_SCATTERED",
        "LINEAR",
        "LINEAR-SEGMENTAL",
        "REGIONAL",
        "REGIONAL-REGIONAL",
        "SEGMENTAL",
    ],
]

#%%

concepts_shape, concepts_margins = concepts
single_concepts_shape = sorted(set(x for c in concepts_shape for x in c.split("-")))
single_concepts_margins = sorted(set(x for c in concepts_margins for x in c.split("-")))
