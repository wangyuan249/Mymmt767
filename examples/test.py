import torch
import numpy as np
from collections import OrderedDict
from operator import itemgetter

if __name__ == '__main__':
    fea = (torch.rand(2048,16,8)* 100).float()
    score = (torch.rand([1,16,8])* 100).float()
    print(score)
    height = score.size(1)
    width = score.size(2)

    my_dict = OrderedDict()

    for i in range(0, height):
        for j in range(0, width):
            # print(score[0,i,j])
            my_dict[score[0,i,j]] = (i,j)

    print(my_dict)
    # my_order_dict = sorted(my_dict.keys(), reverse=True)
    my_order_dict = sorted(my_dict.items(),key=lambda it:it[0],reverse=True)
    print(my_order_dict)
    print(my_order_dict.__class__)
    index = 0

    tensorSelect = torch.empty(0)
    for k, v in my_order_dict:
        if index < 10:
            if index == 0:
                tensorSelect = fea[:, v[0], v[1]]
            else:
                torch.stack()
                tensorSelect = torch.cat((tensorSelect, fea[:, v[0], v[1]]), 1)
        index += 1
        pass

    print(tensorSelect.shape)








