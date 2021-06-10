import numpy as np
from decimal import *
getcontext().prec = 50

para_to_decom_layer = 9*128*128*(2*3+1) +\
                      9*256*256*(2*5+1) +\
                      9*512*512*(2*2+1) +\
                      3*3*64*128 + 3*3*128*256 + 3*3*256*512



def para_of_un_decom_layer():
    num = 7*7*64*3 + 3*3*64*64*6  + 512*1000 + 64*128 +128*256 +256*512 +6+1000
    return num


all_para  = para_to_decom_layer + para_of_un_decom_layer()

def even_ave(rank_rate_ave):
    num_ave = 2*128*(128*rank_rate_ave) + 9*(128*rank_rate_ave)*(128*rank_rate_ave) +\
              2*256*(256*rank_rate_ave) + 9*(256*rank_rate_ave)*(256*rank_rate_ave) +\
              2*512*(512*rank_rate_ave) + 9*(512*rank_rate_ave)*(512*rank_rate_ave)
    return num_ave 


def even_single(rank_rate_single):
    num_single =  (2*128*np.floor(128*rank_rate_single) + 9*np.floor(128*rank_rate_single)*np.floor(128*rank_rate_single))*4 +\
                  (2*256*np.floor(256*rank_rate_single) + 9*np.floor(256*rank_rate_single)*np.floor(256*rank_rate_single))*6 +\
                  (2*512*np.floor(512*rank_rate_single) + 9*np.floor(512*rank_rate_single)*np.floor(512*rank_rate_single))*3 
    return num_single 




def odd_ave(rank_rate_ave):
    num_ave = 2*128*np.floor(128*rank_rate_ave) + 9*np.floor(128*rank_rate_ave)*np.floor(128*rank_rate_ave) +\
              2*256*np.floor(256*rank_rate_ave) + 9*np.floor(256*rank_rate_ave)*np.floor(256*rank_rate_ave) +\
              2*512*np.floor(512*rank_rate_ave) + 9*np.floor(512*rank_rate_ave)*np.floor(512*rank_rate_ave) +\
              64*np.floor(128*rank_rate_ave) + 128*np.floor(256*rank_rate_ave) + 256*np.floor(512*rank_rate_ave)
    return num_ave 


def odd_single(rank_rate_single):
    num_single =  (2*128*np.floor(128*rank_rate_single) + 9*np.floor(128*rank_rate_single)*np.floor(128*rank_rate_single))*3 +\
                  (2*256*np.floor(256*rank_rate_single) + 9*np.floor(256*rank_rate_single)*np.floor(256*rank_rate_single))*5 +\
                  (2*512*np.floor(512*rank_rate_single) + 9*np.floor(512*rank_rate_single)*np.floor(512*rank_rate_single))*2 +\
                  (64*min(np.floor(128*rank_rate_single), 64)) + 128*np.floor(128*rank_rate_single) + 9*min(np.floor(128*rank_rate_single), 64)*np.floor(128*rank_rate_single) +\
                  (128*min(np.floor(256*rank_rate_single), 128)) + 256*np.floor(256*rank_rate_single) + 9*min(np.floor(256*rank_rate_single), 128)*np.floor(256*rank_rate_single) +\
                  (256*min(np.floor(512*rank_rate_single), 256)) + 512*np.floor(512*rank_rate_single) + 9*min(np.floor(512*rank_rate_single), 256)*np.floor(512*rank_rate_single)
    return num_single


def cs_svd(rank_rate_single, rank_rate_ave):
  cs = (even_ave(rank_rate_ave) + even_single(rank_rate_single) + odd_ave(rank_rate_ave) + odd_single(rank_rate_single)+para_of_un_decom_layer())/all_para
  print(cs)

cs_svd(0.3261854184542645,0.25) 
cs_svd(0.4619342907094595,0.25) 
cs_svd(0.7234166524194108,0.25) 
