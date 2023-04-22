from PIL import Image
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import sys


img_name = sys.argv[1]
cut_axis = sys.argv[2]
a = nib.load(img_name).get_fdata()
if cut_axis == "0":
    a = a[a.shape[0] // 2, :, :]
elif cut_axis == "1":
    a = a[:, a.shape[1] // 2, :]
elif cut_axis == "2":
    a = a[:, :, a.shape[2] // 2]
else:
    assert False
print(a.shape)
a[a < -150] = -150
a[a > 250] = 250
a = (a - (-150)) / (250 - (-150))
a = (a * 255).astype(np.uint8)
Image.fromarray(a).save(f'test_axis_{cut_axis}.png')
