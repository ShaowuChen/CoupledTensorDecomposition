import numpy as np
a = np.load('resnet34.npy', allow_pickle=True).item()
for key in a:
    if 'fc' in key:
        print(key)
        print(a[key].shape)