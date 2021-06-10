import numpy as np
import os
from numpy import *
# R=[[1/32,0.24585796],[1/16,0.24088485],[1/8,0.22829500],[1/16,0.49463385],[1/8,0.48737034],[3/16,0.47813796],[1/4,0.46684854],[1/8,0.74106626],[3/16,0.73459623],[1/4,0.72675183]]
R=[[1/16,0.62054625],[1/8,0.6145373],[1/4,0.59768632],[3/8,0.57397854],[1/2,0.54267295]]
def TT_compression(weight, rank_rate):
    F1, F2, I, O = weight.shape  # weight[F1,F2,I,O]
    rank1=int(np.floor(I*rank_rate))
    rank2=int(np.floor(O*rank_rate))
    W = np.reshape(np.transpose(weight, (2, 0, 1, 3)), [I, -1])
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


def initialize_diff(para_dict, r):
    Wd_dict = {}
    Wd_com_dict = {}
    for layer in range(2,5):
        for reapeat in range(2):
            weight_name='layer'+str(layer)+'.'+str(reapeat)+'.conv2' + '.weight'
            Wd_com_dict[weight_name+'.diff.com']=TT_compression(para_dict[weight_name],r)
            Wd_dict[weight_name+'.diff']=TT_compression_restore(Wd_com_dict[weight_name+'.diff.com'])
    return Wd_dict, Wd_com_dict
def iteration_aver(para_dict,Wd_dict, r):
    Wa_dict = {}
    Wa_com_dict = {}
    for layer in range(2,5):
        weight_name_aver='layer'+str(layer)+'.aver'
        w = 0
        for reapeat in range(2):
            weight_name='layer'+str(layer)+'.'+str(reapeat)+'.conv2' + '.weight'
            w1=para_dict[weight_name]-Wd_dict[weight_name+'.diff']
            w+=w1
        Wa_com_dict[weight_name_aver+'.com']=TT_compression(w/2,r)
        Wa_dict[weight_name_aver]=TT_compression_restore(Wa_com_dict[weight_name_aver+'.com'])
    return Wa_dict,Wa_com_dict

def iteration_diff(para_dict,Wa_dict, r):
    Wd_dict = {}
    Wd_com_dict = {}
    for layer in range(2,5):
        for reapeat in range(2):
            weight_name='layer'+str(layer)+'.'+str(reapeat)+'.conv2' + '.weight'
            Wd_com_dict[weight_name+'.diff.com']=TT_compression(para_dict[weight_name]-Wa_dict['layer'+str(layer)+'.aver'],r)
            Wd_dict[weight_name+'.diff']=TT_compression_restore(Wd_com_dict[weight_name+'.diff.com'])
    return Wd_dict, Wd_com_dict
# def iteration_aver(para_dict,Wd_dict,r):
#     Wa_dict = {}
#     Wd_com_dict = {}
#     for layer in range(2,5):
#         weight_name_aver='layer'+str(layer)+'.aver'
#         w = 0
#         for reapeat in range(2):
#             weight_name='layer'+str(layer)+'.'+str(repeat)+'.conv2' + '.weight'
#             w1=para_dict[weight_name]-Wd_dict[weight_name+'.diff']
#             w+=w1
#         Wd_com_dict[weight_name_aver]=TT_compression((w/2+Wa_dict[weight_name_aver])/2,r,r)
#         Wa_dict=TT_compression_restore(Wd_com_dict[weight_name_aver])
#     return Wa_dict,Wd_com_dict


var_dict=np.load('parameter.npy',encoding='latin1',allow_pickle=True).item()

for rank in range(len(R)):
    diff_dict, com_dict_diff = initialize_diff(var_dict, R[rank][1])
    aver_dict ,com_dict_aver= iteration_aver(var_dict, diff_dict, R[rank][0])
    path = './algrithm1_para_npy'
    os.makedirs(path,exist_ok=True)
    # modelP_path1 = os.path.join('', '%s/init_aver_compressed_dict_r1%d_c1%d.npy' % (path, R[rank][0], R[rank][1]))
    # np.save(modelP_path1, com_dict_aver)
    # modelP_path2 = os.path.join('', '%s/init_diff_compressed_dict_r1%d_c1%d.npy' % (path, R[rank][0], R[rank][1]))
    # np.save(modelP_path2, com_dict_diff)
    print("start iteration")
    for i in range(10):
        diff_dict = {}
        com_dict_diff = {}
        diff_dict,com_dict_diff=iteration_diff(var_dict, aver_dict, R[rank][1])
        aver_dict={}
        com_dict_aver={}
        aver_dict,com_dict_aver=iteration_aver(var_dict, diff_dict, R[rank][0])
        print("%dth iteration of r1=%d,c1=%d has finished!!" % (i, R[rank][0], R[rank][1]))
        # if i==4:
        #     modelP_path3 = os.path.join('','%s/iter%d_aver_compressed_dict_r1%d_c1%d.npy' % (path, i, R[rank][0], R[rank][1]))
        #     np.save(modelP_path3, com_dict_aver)
        #     modelP_path4 = os.path.join('','%s/iter%d_diff_compressed_dict_r1%d_c1%d.npy' % (path, i, R[rank][0], R[rank][1]))
        #     np.save(modelP_path4, com_dict_diff)
    modelP_path5 = os.path.join('', '%s/iter%d_aver_compressed_dict_r1=%8f_c1=%8f.npy' % (path, i, R[rank][0], R[rank][1]))
    np.save(modelP_path5, com_dict_aver)
    modelP_path6 = os.path.join('', '%s/iter%d_diff_compressed_dict_r1=%8f_c1=%8f.npy' % (path, i, R[rank][0], R[rank][1]))
    np.save(modelP_path6, com_dict_diff)
