import pydicom
import matplotlib.pyplot as plt
import os
import glob
import pandas as pd

#%%

root_path = "/home/stefano/Datasets/CBIS-DDSM/"
label_file = os.path.join(root_path, "mass_case_description_train_set.csv")
df = pd.read_csv(label_file)

#%%
# All uncropped
all_paths = glob.glob(os.path.join(root_path, "CBIS-DDSM", "*CC", "*", "*"))
all_paths += glob.glob(os.path.join(root_path, "CBIS-DDSM", "*MLO", "*", "*"))
for d in all_paths:
    fs = glob.glob(os.path.join(d, "*"))
    if len(fs) == 0:
        print(f"error on {d}: no files")
    elif len(fs) > 1:
        print(f"error on {d}: more than one file")
    else:
        #print(os.path.join(*os.path.split(fs[0])[:-1], '000000.dcm'))
        os.rename(fs[0], os.path.join(*os.path.split(fs[0])[:-1], '000000.dcm'))


#%%
# All cropped and masked
all_paths = glob.glob(os.path.join(root_path, "CBIS-DDSM", "*CC_*", "*", "*"))
all_paths += glob.glob(os.path.join(root_path, "CBIS-DDSM", "*MLO_*", "*", "*"))
for d in all_paths:
    fs = glob.glob(os.path.join(d, "*"))
    if len(fs) == 0:
        print(f"error on {d}: no files")
    elif len(fs) == 1:
        os.rename(fs[0], os.path.join(*os.path.split(fs[0])[:-1], '000000.dcm'))
    elif len(fs) > 2:
        print(f"error on {d}: more than two files")
    else:
        if os.path.getsize(fs[0]) < os.path.getsize(fs[1]):
            os.rename(fs[0], os.path.join(*os.path.split(fs[0])[:-1], '000000.dcm'))
            os.rename(fs[1], os.path.join(*os.path.split(fs[1])[:-1], '000001.dcm'))
        else:
            os.rename(fs[0], os.path.join(*os.path.split(fs[0])[:-1], '000001.dcm'))
            os.rename(fs[1], os.path.join(*os.path.split(fs[1])[:-1], '000000.dcm'))

#%%

all_paths = glob.glob(os.path.join(root_path, "CBIS-DDSM", "*CC_1", "*", "*"))
all_paths += glob.glob(os.path.join(root_path, "CBIS-DDSM", "*MLO_1", "*", "*"))
