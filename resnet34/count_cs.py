import numpy as np
repeat = [3,4,6,3]
I = [64,64,128,256]
O = [64,128,256,512]




'''计算各种参数个数'''
def all_parameters(print_ph=False):
    num = 7*7*3*64 + 9*64*64*2*3 +\
          9*64*128 + 9*128*128*(2*3+1) +\
          9*128*256 + 9*256*256*(2*5+1) +\
          9*256*512 +9*512*512*(2*2+1) + \
          64 * 128 + 128 * 256 + 256 * 512 +\
          512 * 1000 + 1000
    if print_ph:
        print(num)
    return num

def weight_w_ave(print_ph=False):
    num = 7 * 7 * 4 * 64 + 9 * 64 * 64 * 2 * 3 + \
          9 * 128 * 64 +  9 * 128 * 128 + 9 * 128 * 128 * 2 + \
          9 * 128 * 256 + 9 * 256 * 256 + 9 * 256 * 256 * 2 + \
          9 * 256 * 512 + 9 * 512 * 512 + 9 * 512 * 512 * 2 + \
          64 * 128 + 128 * 256 + 256 * 512 + \
          512 * 1000 + 1000
    if print_ph:
        print(num)

    return num

def weight_decom_single(rank_rate, rank_yellow,print_ph=False):

    num = 7 * 7 * 3 * 64 + 9 * 64 * 64 * 2 * 3 + \
          9 * 64 * 128 + 9  * round(128 * rank_yellow) * round(128 * rank_yellow) + 9 * round(128 * rank_rate) * round(128 * rank_rate) * (2 * 3) + \
          9 * 128 * 256 + 9 * round(256 * rank_yellow) * round(256 * rank_yellow) + 9 * round(256 * rank_rate) * round(256 * rank_rate) * (2 * 5) + \
          9 * 256 * 512 + 9 * round(512 * rank_yellow) * round(512 * rank_yellow) + 9 * round(512 * rank_rate) * round(512 * rank_rate) * (2 * 2) + \
          64 * 128 + 128 * 256 + 256 * 512 + \
          512 * 1000 + 1000
    if print_ph:
        print(num)
    return num


def weight_decom_w_ave(rank_rate_ave, print_ph = False):
    num = 7 * 7 * 3 * 64 + 9 * 64 * 64 * 2 * 3 + \
          9 * 64 * 128 + 9 * 128 * 128 + 9 * round(128 * rank_rate_ave) * round(128 * rank_rate_ave) * 2  + \
          9 * 128 * 256 + 9 * 256 * 256 + 9 * round(256 * rank_rate_ave) * round(256 * rank_rate_ave) * 2  + \
          9 * 256 * 512 + 9 * 512 * 512 + 9 * round(512 * rank_rate_ave) * round(512 * rank_rate_ave) * 2  + \
          64 * 128 + 128 * 256 + 256 * 512 + \
          512 * 1000 + 1000
    if print_ph:
        print(num)
    return num

def weight_other_ave(print_ph=False):
    num =  7 * 7 * 3 * 64 +\
           9 * 64 * 64 * 2 * 3 +\
           9 * 64 * 128 + 9 * 128 * 128 +\
           9 * 128 * 256 + 9 * 256 * 256 +\
           9 * 256 * 512 + 9 * 512 * 512 +\
           64 * 128 + 128 * 256 + 256 * 512 +  512 * 1000 + 1000
    if print_ph:
        print(num)
    return  num

def weight_other_single(print_ph=False):
    num =  7 * 7 * 3 * 64 +\
           9 * 64 * 64 * 2 * 3 +\
           9 * 64 * 128 + \
           9 * 128 * 256 +\
           9 * 256 * 512 + \
           64 * 128 + 128 * 256 + 256 * 512 +  512 * 1000 + 1000
    if print_ph:
        print(num)
    return  num

def weight_yellow_before(print_ph=False):
    num = 9*64*128 + 9*128*256 + 9*128*512
    if print_ph:
        print(num)
    return num



'''求cs'''
def cs_single(rank_rate, rank_yellow,print_ph=False):

    num = weight_decom_single(rank_rate, rank_yellow)/all_parameters()
    if print_ph:
        print(num)
    return num

def cs_ave( print_ph=False):
    sum = weight_w_ave()/all_parameters()
    if print_ph:
        print(sum)
    return sum

def cs_whole_ave_and_single(rank_rate, rank_yellow, print_ph=False):
    weight_w_ave_p =  weight_w_ave()
    weight_decom_single_p = weight_decom_single(rank_rate, rank_yellow)

    all = all_parameters()

    cs = (weight_w_ave_p +weight_decom_single_p-weight_other_single()-weight_yellow_before())/all
    if print_ph:
        print(cs)
    return cs

def cs_decom_both(rank_rate_ave, rank_rate_single, rank_yellow, print_ph=False):
    weight_all = all_parameters()
    weight_decom_w_ave_p = weight_decom_w_ave(rank_rate_ave)
    weight_decom_single_p = weight_decom_single(rank_rate_single, rank_yellow)

    cs = (weight_decom_w_ave_p +weight_decom_single_p-weight_other_single()-weight_yellow_before())/weight_all
    if print_ph:
        print(cs)
    return cs


'''由cs求rank_rate'''
def get_rank_rate_single(cs, rank_rate_yellow, print_ph=False):


    num_parameters = all_parameters()
    temp = cs * num_parameters - (7 * 7 * 3 * 64 + 9 * 64 * 64 * 2 * 3 +\
                                  9 * 64 * 128 + 9  * round(128 * rank_rate_yellow) * round(128 * rank_rate_yellow) +\
                                  9 * 128 * 256 + 9 * round(256 * rank_rate_yellow) * round(256 * rank_rate_yellow) +\
                                  9 * 256 * 512 + 9 * round(512 * rank_rate_yellow) * round(512 * rank_rate_yellow) +\
                                  512 * 1000 + 1000)
    rank_rate = np.sqrt(temp / (9 * ( 128*128*6 + 256*256*10 +512*512*4)))
    if print_ph:
        print(rank_rate)
    return rank_rate

def get_rank_rate_whole_ave_and_single(cs, rank_rate_yellow, print_ph=False):
    num_all_parameters = all_parameters()
    weight_whole_w_ave = weight_w_ave()


    num_deom_without_other = cs * num_all_parameters - weight_whole_w_ave + weight_other_single() + weight_yellow_before() - (7 * 7 * 3 * 64 + 9 * 64 * 64 * 2 * 3 +\
                                  9 * 64 * 128 + 9  * round(128 * rank_rate_yellow) * round(128 * rank_rate_yellow) +\
                                  9 * 128 * 256 + 9 * round(256 * rank_rate_yellow) * round(256 * rank_rate_yellow) +\
                                  9 * 256 * 512 + 9 * round(512 * rank_rate_yellow) * round(512 * rank_rate_yellow) +\
                                  512 * 1000 + 1000)
    rank_rate = np.sqrt(num_deom_without_other/9/(128*128*6 + 256*256*10 + 512*512*4))
    if print_ph:
        print(rank_rate)
    return rank_rate

def get_rank_rate_decom_both(cs, rank_rate_ave,rank_rate_yellow, print_ph=False):


    #                               512 * 1000 + 1000)
    temp = cs * all_parameters() + weight_other_single() + weight_yellow_before() - weight_decom_w_ave(rank_rate_ave) - (7 * 7 * 3 * 64 + 9 * 64 * 64 * 2 * 3 +\
                                  9 * 64 * 128 + 9  * round(128 * rank_rate_yellow) * round(128 * rank_rate_yellow) +\
                                  9 * 128 * 256 + 9 * round(256 * rank_rate_yellow) * round(256 * rank_rate_yellow) +\
                                  9 * 256 * 512 + 9 * round(512 * rank_rate_yellow) * round(512 * rank_rate_yellow) +\
                                  512 * 1000 + 1000)

    temp2 = temp / 9 / (128 * 128 * 6 + 256 * 256 * 10 + 512 * 512 * 4)
    # print(temp2)
    # assert(temp2>0)
    rank_rate = np.sqrt(temp2)
    if print_ph:
        print(rank_rate)
    return rank_rate


all_parameters(print_ph=1)