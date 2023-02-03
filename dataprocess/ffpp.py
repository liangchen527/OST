import os
import sys
import json
from mydataset.distortion import *
import numpy as np
import cv2
import random
from PIL import Image

import torch
from torch.autograd import Variable
from torch.utils import data
from torchvision import transforms as T
import dlib
from skimage import measure
from dataprocess.create_newfake import create_fake
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')


def load_rgb(file_path, size=256):
    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
    return img

class FaceForensicsDataset(data.Dataset):
    data_root = './datasets/FaceForensics++/'
    data_list = {
        'test': './data/FF/test.json',
        'train': './data/FF/train.json',
        'eval': './data/FF/eval.json'
    }

    # frames = {'test': 100, 'eval': 100, 'train': 270}
    frames = {'test': 25, 'eval': 10, 'train': 270}

    def __init__(self, dataset='FF-DF', mode='test', res=256, train=True,
                 sample_num=None, trainset=None):
        self.mode = mode
        self.dataset = dataset
        self.n_sup = 20
        self.n_qry = 16
        self.trainset = trainset
        with open(self.data_list[mode], 'r') as fd:
            data = json.load(fd)
            self.img_lines = []
            self.fake_lines = []
            self.real_lines = []

            for pair in data:
                r1, r2 = pair
                step = 1
                for i in range(0, self.frames[mode], step):

                    self.img_lines.append(('{}/{}'.format('real', r1), i, 0))
                    self.img_lines.append(('{}/{}'.format('real', r2), i, 0))

                    self.real_lines.append(('{}/{}'.format('real', r1), i, 0))
                    self.real_lines.append(('{}/{}'.format('real', r2), i, 0))

                    if self.mode == 'train':
                        if dataset == 'ALL':
                            if i > self.frames[mode] // 4:
                                continue
                            for fake_d in ['FF-DF', 'FF-NT', 'FF-FS', 'FF-F2F']:
                                self.img_lines.append(('{}/{}_{}'.format(fake_d, r1, r2), i, 1))
                                self.img_lines.append(('{}/{}_{}'.format(fake_d, r2, r1), i, 1))

                                self.fake_lines.append(('{}/{}_{}'.format(fake_d, r1, r2), i, 1))
                                self.fake_lines.append(('{}/{}_{}'.format(fake_d, r2, r1), i, 1))
                        else:
                            self.img_lines.append(('{}/{}_{}'.format(dataset, r1, r2), i, 1))
                            self.img_lines.append(('{}/{}_{}'.format(dataset, r2, r1), i, 1))

                            self.fake_lines.append(('{}/{}_{}'.format(dataset, r1, r2), i, 1))
                            self.fake_lines.append(('{}/{}_{}'.format(dataset, r2, r1), i, 1))
                    else:
                        self.img_lines.append(
                            ('{}/{}_{}'.format(dataset, r1, r2), i, 1))
                        self.img_lines.append(
                            ('{}/{}_{}'.format(dataset, r2, r1), i, 1))

        if sample_num is not None:
            self.img_lines = self.img_lines[:sample_num]

        self.img_lines = np.random.permutation(self.img_lines)
        self.fake_lines = np.random.permutation(self.fake_lines)
        self.real_lines = np.random.permutation(self.real_lines)
        normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.transforms = T.Compose([T.ToTensor(), normalize])
        self.totensor = T.Compose([T.ToTensor()])
        self.res = res

    def load_image(self, name, idx):
        impath = '{}/{}/{:04d}.png'.format(self.data_root, name, int(idx))
        img = load_rgb(impath, size=self.res)

        return img

    def __getitem__(self, index):
        if self.mode == 'train':
            name, idx, label = self.img_lines[index]
            img = self.load_image(name, idx)
            rect = detector(img)
            if len(rect) == 0:
                if img is None:
                    print('Img is None!!!')
                else:
                    print('Face detetor failed ...')
                n = np.random.randint(len(self.img_lines))
                o_name, o_idx, o_label = self.img_lines[n]
                o_img = self.load_image(o_name, o_idx)
                aug = o_img
                lab_aug = o_label
            else:
                sp = predictor(img, rect[0])
                i_lmk = np.array([[p.x, p.y] for p in sp.parts()])
                flag = True
                while flag: ########## fake for aug1
                    n = np.random.randint(len(self.img_lines))
                    o_name, o_idx, o_label = self.img_lines[n]
                    o_img = self.load_image(o_name, o_idx)
                    o_rect = detector(o_img)
                    if len(o_rect) == 0:
                        continue
                    else:
                        flag = False
                        o_sp = predictor(o_img, o_rect[0])
                        o_lmk = np.array([[p.x, p.y] for p in o_sp.parts()])
                        aug = create_fake(img, o_img, i_lmk, o_lmk)
                        lab_aug = 1
        else:
            name, idx, label = self.img_lines[index]
            label = int(label)
            img = self.load_image(name, idx)
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
