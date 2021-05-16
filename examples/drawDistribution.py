import matplotlib.pyplot as plt
import numpy as np
import math
import pdb

def dotProduct(v1, v2):
    v1 = np.mat(v1)
    v2 = np.mat(v2)
    z = v1 * v2.T
    return z

if __name__ == '__main__':
    feature = np.load("similarity_list_3.npy")
    print(feature.__class__)
    print(feature.shape)

    similarity_list = []
    feature = feature.reshape(32, 2048, 128)
    # pdb.set_trace()
    # print(feature.shape(2).__class__)
    for k in range(0, feature.shape[0]):
        for i in range(0, feature.shape[2]):
            for j in range(0, feature.shape[2]):
                vector1 = feature[k, :, i]
                vector2 = feature[k, :, j]
                cosine = dotProduct(vector1, vector2) / \
                         math.sqrt(dotProduct(vector1, vector1) * dotProduct(vector2, vector2))
                # print("cosine: ", cosine)
                similarity_list.append(cosine)
        print("k", k)

    similarity_list = np.array(similarity_list).reshape(524288)

    T = np.linspace(0.0, 1.0, num=1000)
    sim_dis = np.zeros(1000)
    for item in similarity_list:
        if(int(item*1000) == 1000):
            continue
        sim_dis[int(item*1000)-1] += 1
    # power = np.array([1.53E+03, 5.92E+02, 2.04E+02, 7.24E+01, 2.72E+01, 1.10E+01, 4.70E+00])
    plt.plot(T,sim_dis)
    # 绘图
    # plt.bar(x=T, height=sim_dis, label='Similarity distribution', color='steelblue', alpha=0.8)
    plt.show()



