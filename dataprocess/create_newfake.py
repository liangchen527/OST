import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from skimage import measure
from skimage.transform import PiecewiseAffineTransform, warp
from dataprocess.utils.face_blend import *
from dataprocess.utils.face_align import get_align_mat_new
from dataprocess.utils.color_transfer import color_transfer
from dataprocess.utils.faceswap_utils import blendImages as alpha_blend_fea
from dataprocess.utils import faceswap

def dist(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def generate_random_mask(mask, res=256):
    randwl = np.random.randint(10, 60)
    randwr = np.random.randint(10, 60)
    randhu = np.random.randint(10, 60)
    randhd = np.random.randint(10, 60)
    newmask = np.zeros(mask.shape)
    mask = np.where(mask > 0.1, 1, 0)
    props = measure.regionprops(mask)
    if len(props) == 0:
        return newmask
    center_x, center_y = props[0].centroid
    center_x = int(round(center_x))
    center_y = int(round(center_y))
    newmask[max(center_x-randwl, 0):min(center_x+randwr, res-1), max(center_y-randhu, 0):min(center_x+randhd, res-1)]=1
    newmask *= mask
    # newmask = random_deform(newmask, 5, 5)
    return newmask

def random_deform(mask, nrows, ncols, mean=0, std=10):
    h, w = mask.shape[:2]
    rows = np.linspace(0, h-1, nrows).astype(np.int32)
    cols = np.linspace(0, w-1, ncols).astype(np.int32)
    #########
    rows += np.random.normal(mean, std, size=rows.shape).astype(np.int32)
    rows += np.random.normal(mean, std, size=cols.shape).astype(np.int32)
    #########
    rows, cols = np.meshgrid(rows, cols)
    anchors = np.vstack([rows.flat, cols.flat]).T
    assert anchors.shape[1] == 2 and anchors.shape[0] == ncols * nrows
    deformed = anchors + np.random.normal(mean, std, size=anchors.shape)
    np.clip(deformed[:,0], 0, h-1, deformed[:,0])
    np.clip(deformed[:,1], 0, w-1, deformed[:,1])

    trans = PiecewiseAffineTransform()
    trans.estimate(anchors, deformed.astype(np.int32))
    warped = warp(mask, trans)
    warped *= mask
    blured = cv2.GaussianBlur(warped, (5, 5), 3)
    # cv2.imwrite('./blurred.png', blured*255)
    return blured

def create_fake(realimg, fakeimg, real_lmk, fake_lmk):
    mask = get_mask(fake_lmk, fakeimg, deform=False) / 255.0
    mask = random_deform(mask, 5, 5)
    newimg,_ = blend_fake_to_real(realimg, real_lmk, fakeimg, fake_lmk, mask)
    return newimg

def blend_fake_to_real(realimg, real_lmk, fakeimg, fake_lmk, deformed_fakemask):
    # deformed_fakemask =random_deform(generate_random_mask(fake_mask), 5, 5)
    # source: fake image
    # taret: real image
    if realimg.max() < 1:
        realimg = (realimg+1)/2 * 255
        fakeimg = (fakeimg+1)/2 * 255
    realimg = realimg.astype(np.uint8)
    fakeimg = fakeimg.astype(np.uint8)
    H, W, C = realimg.shape

    aff_param = np.array(get_align_mat_new(fake_lmk, real_lmk)).reshape(2, 3)
    aligned_src = cv2.warpAffine(fakeimg, aff_param, (W, H),
                                 flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    src_mask = cv2.warpAffine(deformed_fakemask,
                              aff_param, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    src_mask = src_mask > 0  # (H, W)

    tgt_mask = np.asarray(src_mask, dtype=np.uint8)
    tgt_mask = mask_postprocess(tgt_mask)

    ct_modes = ['rct-m', 'rct-fs', 'avg-align', 'faceswap']
    mode_idx = np.random.randint(len(ct_modes))
    mode = ct_modes[mode_idx]

    if mode != 'faceswap':
        c_mask = tgt_mask / 255.
        c_mask[c_mask > 0] = 1
        if len(c_mask.shape) < 3:
            c_mask = np.expand_dims(c_mask, 2)
        src_crop = color_transfer(mode, aligned_src, realimg, c_mask)
    else:
        c_mask = tgt_mask.copy()
        c_mask[c_mask > 0] = 255
        masked_tgt = faceswap.apply_mask(realimg, c_mask)
        masked_src = faceswap.apply_mask(aligned_src, c_mask)
        src_crop = faceswap.correct_colours(masked_tgt, masked_src, np.array(real_lmk))

    type =np.random.randint(2)
    # type = 1
    if tgt_mask.mean() < 0.005 or src_crop.max()==0:
        out_blend = realimg
    else:
        if type == 0:
            out_blend, a_mask = alpha_blend_fea(src_crop, realimg, tgt_mask, featherAmount=0.2 * np.random.rand())
        else:
            b_mask = (tgt_mask * 255).astype(np.uint8)
            l, t, w, h = cv2.boundingRect(b_mask)
            center = (int(l + w / 2), int(t + h / 2))
            out_blend = cv2.seamlessClone(src_crop, realimg, b_mask, center, cv2.NORMAL_CLONE)

    return out_blend, tgt_mask
