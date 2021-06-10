#encoding:utf-8
import numpy as np

 

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
 

def mean_and_std():
    train_dic = unpickle('/home/test01/sambashare/sdh/csw/resnet34/transfer_learning_cifar100/cifar100_dataBase/train')
    train_img = train_dic[b'data']
    train_img = np.transpose(np.reshape(train_img,[50000,3,32,32]),[0,2,3,1])/255

    R = train_img[:,:,:,0]
    G = train_img[:,:,:,1]
    B = train_img[:,:,:,2]


    R_mean = np.mean(R)
    G_mean = np.mean(G)
    B_mean = np.mean(B)
    mean = [R_mean, G_mean, B_mean] #[0.5070751592371322, 0.4865488733149497, 0.44091784336703466]


    R_std = np.std(R)
    G_std = np.std(G)
    B_std = np.std(B)
    std = [R_std, G_std, B_std]# [0.26733428587924063, 0.25643846291708833, 0.27615047132568393]

    return mean,std



def train_shuffle():


    train_dic = unpickle('/home/test01/sambashare/sdh/csw/resnet34/transfer_learning_cifar100/cifar100_dataBase/train')
    train_img = train_dic[b'data']
    train_img = np.transpose(np.reshape(train_img,[50000,3,32,32]),[0,2,3,1])/255

    # mean, std = mean_and_std()
    mean = [0.5070751592371322, 0.4865488733149497, 0.44091784336703466]
    std =  [0.26733428587924063, 0.25643846291708833, 0.27615047132568393]

    train_img = (train_img-mean)/std
    perm = np.random.permutation(50000)
    train_img = train_img[perm]
    train_labels = np.array(train_dic[b'fine_labels'])[perm]

    return train_img, train_labels


def test():

    test_dic = unpickle('/home/test01/sambashare/sdh/csw/resnet34/transfer_learning_cifar100/cifar100_dataBase/test')
    test_img = test_dic[b'data']

    # mean, std = mean_and_std()
    mean = [0.5070751592371322, 0.4865488733149497, 0.44091784336703466]
    std =  [0.26733428587924063, 0.25643846291708833, 0.27615047132568393]

    test_img = np.transpose(np.reshape(test_img,[10000,3,32,32]),[0,2,3,1])/255
    test_img = (test_img-mean)/std
    test_labels = np.array(test_dic[b'fine_labels'])

    return test_img, test_labels


