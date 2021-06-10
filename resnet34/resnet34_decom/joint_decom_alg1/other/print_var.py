import numpy as np

a = value_dict = np.load('./single_tt_decom.npy', allow_pickle=True).item()
for i in a:
    if 'bn' in i and 'Momentum' not in i:
      print(i)
    print(value_dict['layer3.3.bn2/beta'])