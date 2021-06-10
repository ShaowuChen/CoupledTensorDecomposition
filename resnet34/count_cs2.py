import numpy as np
from decimal import *
getcontext().prec = 50

para_to_decom_layer = 9*128*128*(2*3+1) +\
                      9*256*256*(2*5+1) +\
                      9*512*512*(2*2+1)

para_to_ave_layer = 9*128*128*(2+1) +\
                    9*256*256*(2+1) +\
                    9*512*512*(2+1)




def para_of_un_decom_layer():
    num = 7*7*64*3 + 3*3*64*64*6 +\
          3*3*64*128 + 3*3*128*256 + 3*3*256*512 + 512*1000 + 64*128 +128*256 +256*512 +6+1000
    return num



def para_to_singel_decom(rank_rate_single,rank_rate_yellow):

    num_single =  (2*128*(128*rank_rate_single) + 9*(128*rank_rate_single)*(128*rank_rate_single))*6 +\
                  (2*256*(256*rank_rate_single) + 9*(256*rank_rate_single)*(256*rank_rate_single))*10 +\
                  (2*512*(512*rank_rate_single) + 9*(512*rank_rate_single)*(512*rank_rate_single))*4 +\
                  (2*128*(128*rank_rate_yellow) + 9*(128*rank_rate_yellow)*(128*rank_rate_yellow)) +\
                  (2*256*(256*rank_rate_yellow) + 9*(256*rank_rate_yellow)*(256*rank_rate_yellow)) +\
                  (2*512*(512*rank_rate_yellow) + 9*(512*rank_rate_yellow)*(512*rank_rate_yellow))
    return num_single



def para_to_ave_decom_without_first(rank_rate_ave):
    num_ave =     (2*128*(128*rank_rate_ave) + 9*(128*rank_rate_ave)*(128*rank_rate_ave))*2 +\
                  (2*256*(256*rank_rate_ave) + 9*(256*rank_rate_ave)*(256*rank_rate_ave))*2 +\
                  (2*512*(512*rank_rate_ave) + 9*(512*rank_rate_ave)*(512*rank_rate_ave))*2


    return num_ave


def para_to_singel_decom_floor(rank_rate_single,rank_rate_yellow):

    num_single =  (2*128*np.floor(128*rank_rate_single) + 9*np.floor(128*rank_rate_single)*np.floor(128*rank_rate_single))*6 +\
                  (2*256*np.floor(256*rank_rate_single) + 9*np.floor(256*rank_rate_single)*np.floor(256*rank_rate_single))*10 +\
                  (2*512*np.floor(512*rank_rate_single) + 9*np.floor(512*rank_rate_single)*np.floor(512*rank_rate_single))*4 +\
                  (2*128*np.floor(128*rank_rate_yellow) + 9*np.floor(128*rank_rate_yellow)*np.floor(128*rank_rate_yellow)) +\
                  (2*256*np.floor(256*rank_rate_yellow) + 9*np.floor(256*rank_rate_yellow)*np.floor(256*rank_rate_yellow)) +\
                  (2*512*np.floor(512*rank_rate_yellow) + 9*np.floor(512*rank_rate_yellow)*np.floor(512*rank_rate_yellow))
    return num_single



def para_yellow(rank_rate_yellow):

    num =   (2 * 128 * (128 * rank_rate_yellow) + 9 * (128 * rank_rate_yellow) * (128 * rank_rate_yellow)) + \
            (2 * 256 * (256 * rank_rate_yellow) + 9 * (256 * rank_rate_yellow) * (256 * rank_rate_yellow)) + \
            (2 * 512 * (512 * rank_rate_yellow) + 9 * (512 * rank_rate_yellow) * (512 * rank_rate_yellow))

    return num


def para_to_both_decom_floor(rank_rate_single, rank_rate_yellow, ranke_rate_ave):


    return para_to_singel_decom_floor(rank_rate_single,rank_rate_yellow) + para_to_ave_decom_without_first(ranke_rate_ave)


def para_to_both_decom(rank_rate_single, rank_rate_yellow, ranke_rate_ave):


    return para_to_singel_decom(rank_rate_single,rank_rate_yellow) + para_to_ave_decom_without_first(ranke_rate_ave)



def cs_single(rank_rate_single, print_ph=False):
    num = para_to_singel_decom(rank_rate_single,rank_rate_single)/para_to_decom_layer
    if print_ph:
        print('rank_rate_single',num)
    return  num

def cs_both(rank_rate_single, rank_rate_yellow, ranke_rate_ave):
    return para_to_both_decom(rank_rate_single, rank_rate_yellow, ranke_rate_ave)/para_to_decom_layer


def cs_single_all(rank_rate_single, print_ph=False):
    num = (para_to_singel_decom(rank_rate_single,rank_rate_single) + para_of_un_decom_layer()) / (para_to_decom_layer+para_of_un_decom_layer())
    if print_ph==True:
        print(num)
    return num
    
def cs_joint_all(rank_rate_single, rank_rate_yellow, rank_rate_ave, print_ph=False):
    num = (para_to_both_decom_floor(rank_rate_single, rank_rate_yellow, rank_rate_ave) + para_of_un_decom_layer())/(para_to_decom_layer+para_of_un_decom_layer())
    if print_ph==True:
        print(num)
    return num


def count_rank_single(cs, rank_rate_yellow, rank_rate_ave , print_ph=1):



    c = para_to_decom_layer*cs - para_to_ave_decom_without_first(rank_rate_ave) - para_yellow(rank_rate_yellow)
    # c = para_to_decom_layer*cs - para_to_ave_decom_without_first(rank_rate_ave)

    c = -1*c
    a = 9*128*128*6 + 9*256*256*10 + 9*512*512*4
    b = 2*128*128*6 + 2*256*256*10 + 2*512*512*4


    delta=b*b-4*a*c
    # assert(delta>=0)
    if delta<0:
        print('no')
        return 0
    r1 = (-b + np.sqrt(delta))/(2*a)
    r2 = (-b - np.sqrt(delta))/(2*a)
    # print(r1*128)
    # print(r1*256)
    # print(r1*512)
    if print_ph:
        print(r1,' ', r2)
    return r1,r2

# cs1 =  cs_single(3/8,1)
# cs2 =  cs_single(2/4,1)
# cs3 =  cs_single(3/4,1)








# cs_single(3/8,1)
# print('===3/16ave')
# print(cs_both(count_rank_single(cs_single(3/8), 3/8,3/16)[0], 3/8,3/16))
# print('===')
# print(cs_both(count_rank_single(cs_single(3/8), 3/8,1/4)[0], 3/8,1/4))
# print('===')
# print('\n')
# print(cs_both(count_rank_single(cs_single(3/8), 3/8,3/8)[0], 3/8,3/8))
# print('===')
# print('\n')
# print(cs_both(count_rank_single(cs_single(3/8), 3/8,1/2)[0], 3/8,1/2))
# print('===')
# print('\n\n')

# print('================')
# cs_single(2/4,1)
# print(cs_both(count_rank_single(cs_single(1/2), 1/2,1/4)[0], 1/2,1/4))
# print('===')
# print('\n')
# print(cs_both(count_rank_single(cs_single(1/2), 1/2,3/8)[0],  1/2,3/8))
# print('===')
# print('\n')
# print('===')
# print(cs_both(count_rank_single(cs_single(1/2), 1/2,1/2)[0], 1/2,1/2))

# print('\n\n')
# print('================')
# print(cs_single(3/4))
# print(cs_both(count_rank_single(cs_single(3/4), 3/4,1/4)[0],  3/4,1/4))
# print('===')
# print('\n')
# print(cs_both(count_rank_single(cs_single(3/4), 3/4,3/8)[0], 3/4,3/8))
# print('===')
# print('\n')
# print(cs_both(count_rank_single(cs_single(3/4), 3/4,1/2)[0], 3/4,1/2))
# # print('\n')
# print('=======================================')

#
cs_single_all(3/8, 1)
cs_single_all(1/2, 1)
cs_single_all(3/4, 1)
print('=======1=====')
cs_joint_all(0.343830232341034, 3/8, 3/16, print_ph=1)
cs_joint_all(0.3261854184542645, 3/8, 1/4, print_ph=1)
cs_joint_all(0.27720918028416913, 3/8, 3/8, print_ph=1)
cs_joint_all(0.20261683547141943, 3/8, 1/2, print_ph=1)

print('============')



cs_joint_all(0.4619342907094595, 1/2, 1/4, print_ph=1)
cs_joint_all(0.4254943552113008, 1/2, 3/8, print_ph=1)
cs_joint_all(0.3742526722974462, 1/2, 1/2, print_ph=1)

print('============')


cs_joint_all(0.7234166524194108, 3/4, 1/4, print_ph=1)
cs_joint_all(0.6988277643866937, 3/4, 3/8, print_ph=1)
cs_joint_all(0.6658274693443998, 3/4, 1/2, print_ph=1)

# print('===============cout r_i================')
# print(int(np.floor(0.343830232341034*128)))
# print(int(np.floor(0.3261854184542645*128)))
# print(int(np.floor(0.27720918028416913*128)))
# print(int(np.floor(0.20261683547141943*128)))

# print('===============cout r_i================')
# print(int(np.floor(0.4619342907094595*128)))
# print(int(np.floor(0.4254943552113008*128)))
# print(int(np.floor(0.3742526722974462*128)))

# print('===============cout r_i================')
# print(int(np.floor(0.7234166524194108*128)))
# print(int(np.floor(0.6988277643866937*128)))
# print(int(np.floor(0.6658274693443998*128)))


print('===============')
print((para_to_decom_layer+para_of_un_decom_layer()))