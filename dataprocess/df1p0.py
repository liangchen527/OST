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
    if img is None:
        print('empty image at '+file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)

    return img


class DF1P0Dataset(data.Dataset):
    data_root = './datasets/DF1p0_imgs/'
    ff_root = './FaceForensics++/real/'

    data_list = {'test': './test.txt'}

    frames = {'test': 100}#, 'eval': 100, 'train': 270}

    def __init__(self, mode='test', res=256, train=False, trainset=None):
        with open(self.data_list[mode], 'r') as fd:
            data = fd.readlines()
            img_lines = []
            for line in data:
                label = 1 # real:0 fake:1
                name = line[:-1].split('.')[0]
                ffname = name.split('_')[0]
                step = 2
                for i in range(0, self.frames[mode], step):
                    img_lines.append((name, i, label))
                    img_lines.append((ffname, i, 1-label))
                    

        self.img_lines = np.random.permutation(img_lines)
        self.trainset = trainset
        normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.transforms = T.Compose([T.ToTensor(), normalize])
        self.res = res
        self.totensor = T.Compose([T.ToTensor()])

    def load_image(self, name, idx, label):
        if label == 1:
            impath = '{}/{}/{:04d}.png'.format(self.data_root, name, int(idx))
        else:
            impath = '{}/{}/{:04d}.png'.format(self.ff_root, name, int(idx))
        img = load_rgb(impath, size=self.res)
        return img

    def __getitem__(self, index):
        name, idx, label = self.img_lines[index]
        label = int(label)
        img = self.load_image(name, idx, label)
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
        return len(self.img_lines)
