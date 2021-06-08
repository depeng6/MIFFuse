# -*- encoding: utf-8 -*-
"""
@File    : test_RGB.py
@Time    : 2021/6/1 19:58
@Author  : Chool
@Email   : depeng_cust@163.com
@Software: PyCharm
"""
import torch
import numpy as np
from datetime import datetime
import os
from MIF_net import FusionNet
from args_fusion import args
import dataset
from torchvision import utils as vutils
from torch.autograd import Variable
import time
import cv2 as cv

if args.mult_device:
    device_ids = [0, 1]
else:
    device_ids = [0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(path, input_nc, output_nc):
    fusion_model = FusionNet(input_nc, output_nc)

    if args.mult_device:
        fusion_model = torch.nn.DataParallel(fusion_model, device_ids=device_ids)
        fusion_model = fusion_model.cuda(device=device_ids[0])
    else:
        fusion_model = torch.nn.DataParallel(fusion_model)
        fusion_model = fusion_model.cuda()

    fusion_model.load_state_dict(torch.load(path))
    fusion_model.eval()
    para = sum([np.prod(list(p.size())) for p in fusion_model.parameters()])
    type_size = 4
    print('Model {} : params: {:4f}M'.format(fusion_model._get_name(), para * type_size / 1000 / 1000))

    return fusion_model


def generate():

    out_image_path = r"results/fuse_Y"
    mode_path = args.model_path
    in_channel = 1
    out_channel = 1
    total_time = 0.

    if os.path.exists(out_image_path) is False:
        os.makedirs(out_image_path)


    with torch.no_grad():
        model = load_model(mode_path,in_channel,out_channel)
        ir_image_list = dataset.list_images(args.test_rgb_ir_dir)
        vis_image_list = dataset.list_images(args.test_rgb_vis_dir)


        batch_size = args.batch_size_test
        image_set_ir, batches = dataset.load_dataset(ir_image_list, batch_size)
        image_set_vis, batches = dataset.load_dataset(vis_image_list, batch_size)

        for batch in range(batches):

            image_paths_ir = image_set_ir[batch]
            image_paths_vis = image_set_vis[batch]

            img_ir = dataset.get_train_images(image_paths_ir)
            img_vis = dataset.get_train_images(image_paths_vis)


            if args.mult_device:
                img_ir = img_ir.cuda(device=device_ids[0])
                img_vis = img_vis.cuda(device=device_ids[0])
            else:
                img_ir = img_ir.cuda()
                img_vis = img_vis.cuda()

            img_ir = Variable(img_ir, requires_grad=False)
            img_vis = Variable(img_vis, requires_grad=False)
            torch.cuda.synchronize()
            start = time.time()
            out_eval = model(img_ir, img_vis)
            torch.cuda.synchronize()
            end = time.time()

            total_time += end-start

            vutils.save_image(out_eval, os.path.join(out_image_path, f'vi_{batch+1}_Y.png'))
        # print(total_time/10)

def YUV_RGB():
    image_Y_path = r'./results/fuse_Y'
    image_U_path = r'./image/Flir/vis_YUV/U_channel'
    image_V_path = r'./image/Flir/vis_YUV/V_channel'

    img_list_Y = dataset.list_images_UV(image_Y_path)
    img_list_U = dataset.list_images_UV(image_U_path)
    img_list_V = dataset.list_images_UV(image_V_path)
    batch_size = 1
    image_dataset_Y,batch_size = dataset.load_dataset(img_list_Y,batch_size)
    image_dataset_U,_ = dataset.load_dataset(img_list_U,batch_size)
    image_dataset_V,_ = dataset.load_dataset(img_list_V,batch_size)

    merge_image_path = r'./results/Flir/merge/'
    if os.path.exists(merge_image_path) is False:
        os.makedirs(merge_image_path)

    for batch in range(batch_size):
        image_Y_dir = image_dataset_Y[batch]
        image_U_dir = image_dataset_U[batch]
        image_V_dir = image_dataset_V[batch]

        image_Y=cv.imread(image_Y_dir,0)
        image_U=cv.imread(image_U_dir,0)
        image_V=cv.imread(image_V_dir,0)

        image_merge = cv.merge((image_Y,image_U,image_V))
        image_merge = cv.cvtColor(image_merge,cv.COLOR_YUV2RGB)
        cv.imwrite(os.path.join(merge_image_path, f'{batch+1}.png'), image_merge)
    print("Done......")

if __name__ == '__main__':
    generate()
    YUV_RGB()