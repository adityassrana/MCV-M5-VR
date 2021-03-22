import numpy as np
import os, json, random

def cv_split(save_path,folds=4):
    kitti_mots_sequences = np.linspace(0,20,21)
    kitti_test_sequences =np.array([2,6,7,8,10,13,14,16,18]) #from the paper

    kitti_cv_sequences = [int(x) for x in kitti_mots_sequences if x not in kitti_test_sequences]
    mots_sequences = np.array([2,5,9,11])

    np.random.shuffle(kitti_cv_sequences)
    np.random.shuffle(mots_sequences)

    length_kitti_fold = int(len(kitti_cv_sequences)/folds)
    kitti_folds=[]
    for i in range(length_kitti_fold+1):
        kitti_folds.append(kitti_cv_sequences[i*length_kitti_fold:i*length_kitti_fold+length_kitti_fold])

    for k in range(folds):
        f = open(save_path+"val_{}.txt".format(k), "a+")
        for i in range(len(kitti_folds[k])):
            f.write("/home/mcv/datasets/KITTI-MOTS/training/image_02/{}/\n".format(str(kitti_folds[k][i]).zfill(4)))
        f.write("/home/mcv/datasets/MOTSChallenge/train/images/{}/\n".format(str(mots_sequences[k]).zfill(4)))
        f.close()
    
        for j in range(len(kitti_folds)):
            if j!=k:
                f = open(save_path+"train_{}.txt".format(k), "a+")
                for l in range(len(kitti_folds[j])):
                    f.write("/home/mcv/datasets/KITTI-MOTS/training/image_02/{}/\n".format(str(kitti_folds[j][l]).zfill(4)))
                f.write("/home/mcv/datasets/MOTSChallenge/train/images/{}/\n".format(str(mots_sequences[j]).zfill(4)))
                f.close()

cv_split("/home/group02/week3/data/cros_val/")