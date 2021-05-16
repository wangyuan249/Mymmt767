import os
from random import sample
import shutil

filePath = r'D:\dataset\market1501\market1501\raw\Market-1501-v15.09.15\bounding_box_train'
newfilePath = r'D:\dataset\market1501_pick'

filesname = os.listdir(filePath)
file = filesname[0]
class_list = []
pick_num = 5
picknames_list = []
filename_l = filesname[0]
for filename in filesname:
    filename_n = filename
    if filename_n[:4] == filename_l[:4]:
        class_list.append(filename_n)
    else:
        if len(class_list) >= 5:
            picknames_list = sample(class_list,5)
        else:
            picknames_list = class_list
        class_list = []
        # print(picknames_list)
        # temp = 1
        for pickname in picknames_list:
            old_file = filePath + '\\' + pickname
            new_file = newfilePath + '\\' + pickname
            shutil.copy(old_file, new_file)

    filename_l = filename

