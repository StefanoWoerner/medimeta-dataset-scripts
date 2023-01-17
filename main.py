import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

#%%
img = nib.load('/home/stefano/Datasets/medmnist_sources/organ/LITS/Training Batch 1/volume-0.nii')
seg = nib.load('/home/stefano/Datasets/medmnist_sources/organ/LITS/Training Batch 1/segmentation-0.nii')


img_fdata = img.get_fdata()
seg_fdata = seg.get_fdata()

#%%
plt.figure(figsize=(40,40))
for i in range(img.shape[2]):
    plt.subplot(int(img.shape[2]**0.5+1), int(img.shape[2]**0.5+1), i + 1)
    plt.imshow(seg_fdata[:,:,i])
    #plt.gcf().set_size(10, 10)
plt.show()

#%%
s = set()
for vol_num in range(28):
    seg = nib.load(f'/home/stefano/Datasets/medmnist_sources/organ/LITS/Training Batch 1/segmentation-{vol_num}.nii')
    seg_fdata = seg.get_fdata()
    s = s.union(set(x for x in seg_fdata.flatten()))
