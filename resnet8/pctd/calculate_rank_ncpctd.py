import numpy as  np 

#resnet8
in_out = [[64,128],[128,128],[128,256],[256,256],[256,512],[512,512]]
R_tt_str = ['1/4', '3/8', '1/2', '5/8', '3/4']
R_c_str = ['1/8','3/16', '1/4', '3/8', '1/2']

R_tt = [eval(i) for i in R_tt_str]
R_c = [eval(i) for i in R_c_str]






def other_parameter():
    num = 3*3*3*64 + 64*128 +128*256 +256*512
    return num

#num of  parameters of original resnet8
def orig():
    num = 0
    for i in range(len(in_out)):
        num += 3*3*in_out[i][0]*in_out[i][1] 
    return num 

def tt(rank_tt):

    #以小的维 为基准计算rank  
    rank = [int(in_out[i][0]*rank_tt) for i in range(len(in_out))]

    num = 0 
    for i in range(len(in_out)):
        num += in_out[i][0]*rank[i] + 9 * rank[i] * rank[i] + rank[i] * in_out[i][1]

    return num 


# def tt(rank_tt):

#     rank = [[int(in_out[i][0]*rank_tt),int(in_out[i][1]*rank_tt)] for i in range(len(in_out))]
#     num = 0 
#     for i in range(len(in_out)):
#         num += in_out[i][0]*rank[i][0] + 9 * rank[i][0] * rank[i][1] + rank[i][1] * in_out[i][1]

 
#     return num 




#已知rank_tt， r_c， 求独立分量的r_i
def count_pctd_ri(rank_tt, r_c, print_ph=0):
    num_tt = tt(rank_tt)

    #为小维度为基准计算r1/r2，即r1==r2==rc
    rank_c  = [int(in_out[i][0] * r_c) for i in range(len(in_out))]

    num_com_para = 0
    for i in range(3):
        i = i * 2  #0,2,4

        num_com_para += (in_out[i][0] + in_out[i][1]) * rank_c[i] + 9 * rank_c[i] * rank_c[i] + rank_c[i] * in_out[i][1]

    num_indep_para = tt(rank_tt) - num_com_para

    #方程
    #ranki
    # num_indep_para = sum( i * (i *ri) + 9*(i*ri)* (i * ri） + o* (i * ri)) = sum( 9*i*i ) * (ri^2) + sum( (i*i + i*o)) * ri 
    #即解一元二次方程 a = sum( 9*i*i ), b = sum( (i*i + i*o)), c = -  num_indep_para
    a = 0
    for i in range(len(in_out)):
        a += 9 * in_out[i][0] * in_out[i][0]
    
    b = 0
    for i in range(len(in_out)):
        b += in_out[i][0] * in_out[i][0] + in_out[i][0] * in_out[i][1]

    c = -num_indep_para

    delta=b*b-4*a*c
    # assert(delta>=0)
    if delta<0:
        # print('no')
        return None,None
    r1 = (-b + np.sqrt(delta))/(2*a)
    r2 = (-b - np.sqrt(delta))/(2*a)

    if print_ph:
        print(r1,' ', r2)
    return r1,r2



def count_para_pctd(r_c, r_i):

    #为小维度为基准计算r1/r2，即r1==r2==rc
    rank_c  = [int(in_out[i][0] * r_c) for i in range(len(in_out))]

    num_com_para = 0
    for i in range(3):
        i = i * 2  #0,2,4

        num_com_para += (in_out[i][0] + in_out[i][1]) * rank_c[i] + 9 * rank_c[i] * rank_c[i] + rank_c[i] * in_out[i][1]


    # num_com_para = 0
    # #为小维度为基准计算r1/r2，即r1==r2==rc
    # rank_c  = [int(in_out[i][0] * r_c) for i in range(len(in_out))]
    # for i in range(len(in_out)):
    #     num_com_para += (in_out[i][0] + in_out[i][1]) * rank_c[i] + 9 * rank_c[i] * rank_c[i] + rank_c[i] * in_out[i][1]


    # a = 0
    # for i in range(len(in_out)):
    #     a += 9 * in_out[i][0] * in_out[i][0]
    
    # b = 0
    # for i in range(len(in_out)):
    #     b += in_out[i][0] * in_out[i][0] + in_out[i][0] * in_out[i][1]            
            
    # num_indep_para = int(a*r_i*r_i + b*r_i)

    a = 0
    for i in range(len(in_out)):
        a += 9 * int(in_out[i][0] *r_i) * int(in_out[i][0]*r_i)
    
    b = 0
    for i in range(len(in_out)):
        b += in_out[i][0] * int(in_out[i][0]*r_i) + int(in_out[i][0] *r_i)* in_out[i][1]            
            
    num_indep_para = a + b

    num = num_indep_para + num_com_para

    return num



R_i = []
for r_tt in R_tt:
    ri= []
    for r_c in R_c:

        r1, r2 = count_pctd_ri(r_tt, r_c)
        print('rank_tt r_c r1 r2: ', r_tt, r_c, r1, r2)

        ri.append(r1)
    R_i.append(ri)

print('R——tt')
print(R_tt_str)
print((np.array([eval(i) for i in R_tt_str])*128).astype(int))


print('\n对应的独立分量的秩：')
for i in range(5):
    print('R_tt='+str(R_tt_str[i]) + ' R_c=' +str(R_c_str))
    print(R_i[i])

print('\nR_i')
print(R_i)
print((np.array(R_i)*128).astype(int))

print('\nR_c')
print(R_c)
print(np.array(R_c)*128)

para_pctd = []
for i in range(5):
    Num = []
    for j in range(5):
        num = count_para_pctd(R_c[j],R_i[i][j])
        Num.append(num)
    para_pctd.append(Num)




para_tt = []
for i in range(len(R_tt)):
    para_tt.append(tt(R_tt[i]))

print('\npara_tt:',para_tt)
print('para_pctd:',para_pctd)

R_c_8f =  ['%8f'%i for i in R_c]
print('R_c_8f', R_c_8f)
R_i_8f = [[eval('%8f'%i) for i in j] for j in R_i]
print('R_i_8f', R_i_8f)


para_tt = np.array(para_tt)
para_pctd = np.array(para_pctd)
num_other_parameter = other_parameter()

print('\nRatio')
print((para_tt+num_other_parameter)/(orig()+num_other_parameter))
print((para_pctd+num_other_parameter)/(orig()+num_other_parameter))


print('\nCR')
print(1/((para_tt+num_other_parameter)/(orig()+num_other_parameter)))
print(1/((para_pctd+num_other_parameter)/(orig()+num_other_parameter)))

# print(other_parameter())
# print(orig())

#ncptd 结果
r_q = np.array([3/8,1/2, 5/8])*128
print(r_q)

print(orig()+num_other_parameter)