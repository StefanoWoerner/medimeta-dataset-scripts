import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io

#%%
img = io.imread('/home/stefano/Datasets/LUNA/subset0/1.3.6.1.4.1.14519.5.2.1.6279.6001.105756658031515062000744821260.mhd', plugin='simpleitk')

#%%
plt.figure(figsize=(40,40))
for i in range(img.shape[0]):
    plt.subplot(int(img.shape[0]**0.5+1), int(img.shape[0]**0.5+1), i + 1)
    plt.imshow(img[i,:,:])
    #plt.gcf().set_size(10, 10)
plt.show()