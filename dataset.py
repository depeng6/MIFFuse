# -*- encoding: utf-8 -*-
"""
@File    : dataset.py
@Time    : 2021/3/24 20:50
@Author  : Chool
@Email   : depeng_cust@163.com
@Software: PyCharm
"""

import os
from os import listdir
from os.path import join
import numpy as np
import torch
from PIL import Image
from args_fusion import args
from scipy.misc import imread
import cv2 as cv
from torchvision import  transforms
from torch.utils.data import Dataset
from torchvision import utils as vutils

def list_images(directory):
    images = []
    names = []
    dir = listdir(directory)
    dir.sort(key=lambda x:int(x[3:-4]))
    for file in dir:
        name = file.lower()
        if name.endswith('.png'):
            images.append(join(directory, file))
        if name.endswith('.bmp'):
            images.append(join(directory, file))
        elif name.endswith('.jpg'):
            images.append(join(directory, file))
        elif name.endswith('.jpeg'):
            images.append(join(directory, file))
        elif name.endswith('.tif'):
            images.append(join(directory, file))
        name1 = name.split('.')
        names.append(name1[0])
    return images


# load training images
def load_dataset(image_path, BATCH_SIZE, num_imgs=None):
    if num_imgs is None:
        num_imgs = len(image_path)
    original_imgs_path = image_path[:num_imgs]

    batches = int(len(original_imgs_path) // BATCH_SIZE)
    return original_imgs_path, batches


def get_train_images(paths):
    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        image = imread(path)
        if image.ndim == 3:
            image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        image = np.expand_dims(image,0)
        image = image.astype(np.float32) / 255.0
        images.append(image)

    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).float()
    return images


def rand(a=0, b=1):
        return np.random.rand() * (b - a) + a

class Fusion_Dataset(Dataset):
    def __init__(self,ir_lines,vis_lines,gt_lines,random_data=True):
        super(Fusion_Dataset,self).__init__()

        self.ir_lines = ir_lines
        self.vis_lines = vis_lines
        self.gt_lines = gt_lines
        self.train_batches = len(ir_lines)
        self.random_data = random_data

    def rand(self,a=0, b=1):
        return np.random.rand() * (b - a) + a

    def __len__(self):
        return self.train_batches

    def get_random_data(self,image_ir,image_vis,image_gt):
        image_ir = np.reshape(image_ir, [1, image_ir.shape[0], image_ir.shape[1]])
        image_vis = np.reshape(image_vis, [1, image_vis.shape[0], image_vis.shape[1]])
        image_gt = np.reshape(image_gt, [1, image_gt.shape[0], image_gt.shape[1]])
        flip = rand(self)<0.5
        if flip:
            image_ir = cv.flip(image_ir,1)
            image_vis = cv.flip(image_vis,1)
            image_gt = cv.flip(image_gt,1)

        image_ir = np.asarray(image_ir, np.float32)
        image_vis = np.asarray(image_vis, np.float32)
        image_gt = np.asarray(image_gt, np.float32)

        image_ir = image_ir.astype(np.float32) / 255.0
        image_vis = image_vis.astype(np.float32) / 255.0
        image_gt = image_gt.astype(np.float32) / 255.0

        image_ir = torch.from_numpy(image_ir).float()
        image_vis = torch.from_numpy(image_vis).float()
        image_gt = torch.from_numpy(image_gt).float()

        return image_ir,image_vis,image_gt

    def __getitem__(self, index):
        ir_name = self.ir_lines[index]
        ir_name = ir_name.split()[0]
        vis_name = self.vis_lines[index]
        vis_name = vis_name.split()[0]
        gt_name = self.gt_lines[index]
        gt_name = gt_name.split()[0]


        image_ir = Image.open(args.train_ir_dir+'/'+ir_name+'.jpg').convert("L")
        image_vis = Image.open(args.train_vis_dir+'/'+vis_name+'.jpg').convert("L")
        image_gt = Image.open(args.train_gt_dir+'/'+gt_name+'.jpg').convert("L")

        trans = transforms.Compose([transforms.ToTensor()])
        image_ir = trans(image_ir)
        image_vis = trans(image_vis)
        image_gt = trans(image_gt)

        return image_ir,image_vis,image_gt


def save_valid_image(image,epoch,batch):
    save_path = "valid_image/{}/Epoch_{}".format(args.model_name,epoch)
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    vutils.save_image(image,os.path.join(save_path, 'fusion_'f'{batch}.png'))











#----------------在真实测试红外与可见光图像融合时使用-------------------------------
# class Fusion_Dataset(Dataset):
#     def __init__(self,ir_lines,vis_lines,random_data=True):
#         super(Fusion_Dataset,self).__init__()
#
#         self.ir_lines = ir_lines
#         self.vis_lines = vis_lines
#         self.train_batches = len(ir_lines)
#         self.random_data = random_data
#
#     def rand(self,a=0, b=1):
#         return np.random.rand() * (b - a) + a
#
#     def __len__(self):
#         return self.train_batches
#
#     def get_random_data(self,image_ir,image_vis):
#
#         # 其中1 指的是通道为单通道
#         # todo
#         image_ir = np.reshape(image_ir, [1, image_ir.shape[0], image_ir.shape[1]])
#         image_vis = np.reshape(image_vis, [1, image_vis.shape[0], image_vis.shape[1]])
#         # print(image_ir.shape)
#
#         # flip image or not
#         flip = rand(self)<0.5
#         if flip:
#             # image_ir = image_ir.transpose(Image.FLIP_LEFT_RIGHT)
#             # image_vis = image_vis.transpose(Image.FLIP_LEFT_RIGHT)
#             image_ir = cv.flip(image_ir,0)
#             image_vis = cv.flip(image_vis,0)
#
#         image_ir = np.asarray(image_ir, np.float32)
#         image_vis = np.asarray(image_vis, np.float32)
#         image_ir = image_ir.astype(np.float32) / 255.0
#         image_vis = image_vis.astype(np.float32) / 255.0
#
#         # print("image.shape ", image_ir.shape)
#
#         image_ir = torch.from_numpy(image_ir).float()
#         image_vis = torch.from_numpy(image_vis).float()
#
#         return image_ir,image_vis
#
#     def __getitem__(self, index):
#         ir_name = self.ir_lines[index]
#         ir_name = ir_name.split()[0]
#         vis_name = self.vis_lines[index]
#         vis_name = vis_name.split()[0]
#         # print(ir_name)
#
#         # print(name)
#         # 从文件读取图像
#         # todo
#         # image_ir = cv.imread(r"./image/IR"+'/'+ir_name+'.png')
#         # image_vis = cv.imread(r"./image/VIS"+'/'+vis_name+'.png')
#
#         image_ir = cv.imread(r"./image/ir_y"+'/'+ir_name+'.png')
#         image_vis = cv.imread(r"./image/vis_y"+'/'+vis_name+'.png')
#
#         image_ir = cv.cvtColor(image_ir,cv.COLOR_RGB2GRAY)
#         image_vis = cv.cvtColor(image_vis,cv.COLOR_RGB2GRAY)
#         # print("image.shape ", image_ir.shape)
#
#         if self.random_data:
#             image_ir,image_vis = self.get_random_data(image_ir,image_vis)
#
#         return image_ir,image_vis

