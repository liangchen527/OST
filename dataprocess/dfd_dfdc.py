import os
import sys
import json

import numpy as np
import cv2
import random
from PIL import Image
import torch
from torch.autograd import Variable
from torch.utils import data
from torchvision import transforms as T
import torchvision
import dlib
from dataprocess.create_newfake import create_fake
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')


def load_rgb(file_path, size=256):
    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)

    return img

class DFD_DFDCDataset(data.Dataset):
    def __init__(self, data_list_file, res=256, train=True, trainset = None):
        self.res = res
        self.dfdc = 0
        self.trainset = trainset
        if 'DFDC/' in data_list_file:
            self.data_root = './datasets/DFDC'
            self.dfdc = 1
        else:
            self.data_root = './datasets/DFD'

        with open(data_list_file, 'r') as fd:
            imglines = fd.readlines()

        self.imglines = np.random.permutation(imglines)
        normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.transforms = T.Compose([T.ToTensor(), normalize])
        

    def __getitem__(self, index):
        sample = self.imglines[index]
        splits = sample.split(',')

        if len(splits) == 1:
            print('the len(splits) =1 : str ='.format(splits))

        img_path = splits[0] # os.path.join(self.data_root, splits[0])

        label = int(splits[1])
        if self.dfdc:
            img_path = self.data_root + img_path.split('DFDC')[1]
        else:
            img_path = self.data_root + img_path.split('DFD')[1]
        img = load_rgb(img_path, self.res)
        ##############
        ##############
        rect = detector(img)
        if len(rect) == 0:
            if img is None:
                print('Img is None!!!')
            else:
                print('Face detetor failed ...')
            n = np.random.randint(len(self.trainset.img_lines))
            o_name, o_idx, o_label = self.trainset.img_lines[n]
            o_img = self.trainset.load_image(o_name, o_idx)
            aug = o_img
            lab_aug = o_label
        else:
            sp = predictor(img, rect[0])
            i_lmk = np.array([[p.x, p.y] for p in sp.parts()])
            flag = True
            while flag:  ########## fake for aug1
                n = np.random.randint(len(self.trainset.img_lines))
                o_name, o_idx, o_label = self.trainset.img_lines[n]
                o_img = self.trainset.load_image(o_name, o_idx)
                o_rect = detector(o_img)
                if len(o_rect) == 0:
                    continue
                else:
                    flag = False
                    o_sp = predictor(o_img, o_rect[0])
                    o_lmk = np.array([[p.x, p.y] for p in o_sp.parts()])
                    aug = create_fake(img, o_img, i_lmk, o_lmk)
                    lab_aug = 1

        img = self.transforms(Image.fromarray(np.array(img, dtype=np.uint8)))
        aug = self.transforms(Image.fromarray(np.array(aug, dtype=np.uint8)))
        o_img = self.transforms(Image.fromarray(np.array(o_img, dtype=np.uint8)))
        label = int(label)
        lab_aug = int(lab_aug)
        o_label = int(o_label)

        return img, label, aug, lab_aug, o_img, o_label

    def __len__(self):
        return len(self.imglines)


