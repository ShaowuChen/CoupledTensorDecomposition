import numpy as np
import math
value_dict = np.load('./single_tt_decom.npy', allow_pickle=True).item()
for i in value_dict:
    if 'layer2.2' in i and 'weight' in  i and 'Momentum' not in i:
        print(i)
W = value_dict['layer2.2.conv1.weight']
W1 = value_dict['layer2.2.conv1.weight1']
W2 = value_dict['layer2.2.conv1.weight2']
W3 = value_dict['layer2.2.conv1.weight3']

print('shape of W: ',W.shape)
print('shape of W1: ',W1.shape)
print('shape of W2: ',W2.shape)
print('shape of W3: ',W3.shape)

u1 = np.reshape(W1,[128,128])
v1 = np.reshape(np.transpose(W2,[2,0,1,3]),[128,3*3*128])
v2 = np.reshape(W3, [128,128])

G12_2d = np.matmul(u1,v1)
g12_2d = np.reshape(np.transpose(np.reshape(G12_2d,[128,3,3,128]),[1,2,0,3]),[3*3*128,128])

W_2d = np.matmul(g12_2d, v2 )
W_re = np.reshape(W_2d, [3,3,128,128])

print(np.sum(W_re - W))
print(1024*458+9*458*458-9*512*512)
# 9r^2+ (I+O)r = （I*O*9)*rank_rate
print(3*3*512*512-458*(512+512)-9*458*458)
print(3*3*512*512-512*(512+512)-9*512*512)


a = 9
I = 512
O = 512
rank_rate = 1
b = I + O
c = -I * O * 9 * rank_rate
delta = b * b - 4 * a * c

print(delta)
rank_1 = (-b + math.sqrt(delta)) / (2 * a)
rank_2 = (-b - math.sqrt(delta)) / (2 * a)
print(rank_1)
print(rank_2)
