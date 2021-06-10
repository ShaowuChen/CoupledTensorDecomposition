import numpy as np
import os
import tensorflow as tf
from numpy import *

R=[[1/4,0.609375]]
def TT_compression(weight, rank_rate):
    F1, F2, I, O = weight.shape  # weight[F1,F2,I,O]
    rank1=int(np.floor(I*rank_rate))
    rank2=int(np.floor(O*rank_rate))
    W = np.reshape(np.transpose(weight,(2,0,1,3)), [I, -1])
    u1, s1, v1 = np.linalg.svd(W, full_matrices=True)
    G1  = np.dot(u1[:, 0:rank1].copy(),np.diag(s1)[0:rank1, 0:rank1].copy())
    G1 = np.reshape(G1, [1, 1, I, rank1])
    V1 = v1[0:rank1, :].copy()
    V1 = np.reshape(np.reshape(V1, [rank1,F1,F1,O]),[-1, O])
    u2, s2, v2 = np.linalg.svd(V1, full_matrices=True)
    G2 = np.dot(u2[:, 0:rank2].copy(),np.diag(s2)[0:rank2, 0:rank2].copy())
    G2= np.transpose(np.reshape(G2, [rank1, 3, 3, rank2]), (1, 2, 0, 3))
    G3 = v2[0:rank2, :].copy()
    G3 = G3.reshape((1, 1, rank2, O))
    return [G1, G2, G3]


def TT_compression_restore(dict):
    G1 = dict[0]
    G2 = dict[1]
    G3 = dict[2]
    m1, n1, j1, k1 = G1.shape
    m2, n2, j2, k2 = G2.shape
    m3, n3, j3, k3 = G3.shape
    rank1=k1
    rank2=k2
    I=j1
    O=k3
    G2 = np.reshape(np.transpose(G2,(2,0,1,3)), [-1, rank2])
    G3 = np.reshape(G3,[rank2, -1])
    G23 = np.reshape(np.reshape(np.dot(G2, G3), [rank1,3,3,O]), [rank1,-1])
    G1 = np.reshape(G1, [I, rank1])
    W = np.transpose(np.reshape(np.dot(G1, G23), [I, 3,3,O]),(1,2,0,3))
    return W


def iteration_f1(para_dict,f3_dict,r):
    f1_dict={}
    com_dict={}
    # 此循环为了得到未切片的P列表,以及每个block共享的Q,H列表
    for layer in range(2,5):
        W_list=[]
        shape_list = []
        min_Channel=para_dict['layer' + str(layer) + '.0' + '.conv1' + '.weight'].shape[2]
        max_channel=para_dict['layer' + str(layer) + '.0' + '.conv1' + '.weight'].shape[3]
        for repeat in range(2):
            weight_name = 'layer' + str(layer) + '.' + str(repeat) + '.conv1' + '.weight'
            w=np.reshape(para_dict[weight_name]-f3_dict[weight_name+'.f3'],[(para_dict[weight_name]-f3_dict[weight_name+'.f3']).shape[2],-1])
            W_list.append(w)
            shape_list.append(para_dict[weight_name].shape)
        W=np.vstack((W_list[0],W_list[1]))
        u1,s1,v1= np.linalg.svd(W, full_matrices=True)
        P = np.dot(u1[:, 0:int(np.floor((min_Channel*r)))].copy(), np.diag(s1)[0:int(np.floor(min_Channel*r)), 0:int(np.floor(min_Channel*r))].copy())
        P1_1=P[0:shape_list[0][2],:]
        P1=np.reshape(P1_1,[1,1,shape_list[0][2],int(np.floor(min_Channel*r))])
        P2_1=P[shape_list[0][2]:,:].reshape((1,1,shape_list[1][2],int(np.floor(min_Channel*r))))
        P2 = np.reshape(P2_1, [1, 1, shape_list[1][2], int(np.floor(min_Channel * r))])
        V1 = v1[0:int(np.floor(min_Channel*r)), :].copy()
        V1=np.reshape(np.reshape(V1,[int(np.floor(min_Channel*r)),(para_dict[weight_name]-f3_dict[weight_name+'.f3']).shape[1],(para_dict[weight_name]-f3_dict[weight_name+'.f3']).shape[1],(para_dict[weight_name]-f3_dict[weight_name+'.f3']).shape[3]]),[-1,(para_dict[weight_name]-f3_dict[weight_name+'.f3']).shape[3]])
        u2, s2, v2 = np.linalg.svd(V1, full_matrices=True)
        Q1 = np.dot(u2[:, 0:int(np.floor((max_channel*r)))].copy(), np.diag(s2)[0:int(np.floor((max_channel*r))), 0:int(np.floor((max_channel*r)))].copy())
        Q=np.transpose(np.reshape(Q1,[int(np.floor(min_Channel*r)),3,3,int(np.floor(max_channel*r))]),(1,2,0,3))
        H1 = v2[0:int(np.floor((max_channel*r))), :].copy()
        H=H1.reshape((1,1,int(np.floor((max_channel*r))),shape_list[0][3]))
        com_dict['layer'+str(layer)+'.f1.com']=[P1,P2,Q,H]
        w1 = TT_compression_restore([P1,Q,H])
        w2=TT_compression_restore([P2,Q,H])
        f1_dict['layer' + str(layer) + '.f1']=[w1,w2]

    return f1_dict,com_dict
def initialize_f3(para_dict, r):
    f3_dict={}
    f3_com_dict={}
    for layer in range(2, 5):
        for reapeat in range(2):
            weight_name = 'layer' + str(layer) + '.' + str(reapeat) + '.conv1' + '.weight'
            f3_com_dict[weight_name + '.f3.com'] = TT_compression(para_dict[weight_name], r)
            f3_dict[weight_name + '.f3'] = TT_compression_restore(f3_com_dict[weight_name + '.f3.com'])
    return f3_dict, f3_com_dict

def iteration_f3(para_dict, f1_dict, r):
    f3_dict={}
    com_dict={}
    for layer in range(2,5):
        for reapeat in range(2):
            weight_name = 'layer' + str(layer) + '.' + str(reapeat) + '.conv1' + '.weight'
            com_dict[weight_name + '.f3.com'] = TT_compression(para_dict[weight_name]-f1_dict['layer'+str(layer)+ '.f1'][reapeat], r)
            f3_dict[weight_name + '.f3'] = TT_compression_restore(com_dict[weight_name + '.f3.com'])
    return f3_dict,com_dict

var_dict=np.load('parameter.npy',encoding='latin1',allow_pickle=True).item()

for rank in range(len(R)):
    f3, com_f3 = initialize_f3(var_dict, R[rank][1])
    f1 ,com_f1= iteration_f1(var_dict, f3, R[rank][0])
    path = './algrithm2_npy'
    os.makedirs(path,exist_ok=True)
    print("start iteration")
    for i in range(10):
        f3 = {}
        com_f3 = {}
        f3, com_f3=iteration_f3(var_dict, f1, R[rank][1])
        f1={}
        com_f1={}
        f1, com_f1=iteration_f1(var_dict, f3, R[rank][0])
        print("%dth iteration of r1=%d,c2=%d has finished!!" % (i, R[rank][0], R[rank][1]))

    modelP_path1 = os.path.join('', '%s/iter%d_f1_compressed_dict_r1=%8f_c2=%8f.npy' % (path, i,R[rank][0], R[rank][1]))
    np.save(modelP_path1, com_f1)
    modelP_path2 = os.path.join('', '%s/iter%d_f3_compressed_dict_r1=%8f_c2=%8f.npy' % (path, i, R[rank][0], R[rank][1]))
    np.save(modelP_path2, com_f3)








