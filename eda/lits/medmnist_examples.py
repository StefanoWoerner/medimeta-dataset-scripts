import os
import numpy as np
import PIL.Image as Image
from medmnist import INFO

n_examples = 10

for view in ('a', 's', 'c'):
    dataset = f'organ{view}mnist'
    os.makedirs(dataset, exist_ok=True)
    labels = INFO[dataset]['label']
    data = np.load(f'{dataset}.npz')
    x = data['train_images']
    y = data['train_labels'].flatten()
    for label in labels.items():
        lab_i = int(label[0])
        exs = x[y == lab_i][:n_examples]
        for i, ex in enumerate(exs):
            Image.fromarray(ex).save(f'{dataset}/{label[1]}_{i}.tiff')
