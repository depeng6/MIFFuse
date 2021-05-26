# -*- encoding: utf-8 -*-
"""
@File    : test_model.py
@Time    : 2021/4/18 19:32
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


def main():

    TIME_STAMP = "{0:%Y-%m-%dT-%H-%M-%S/}".format(datetime.now())
    out_image_path = r"results/model_{}".format(TIME_STAMP)
    mode_path = args.model_path
    in_channel = 1
    out_channel = 1
    total_time = 0.

    if os.path.exists(out_image_path) is False:
        os.makedirs(out_image_path)


    with torch.no_grad():
        model = load_model(mode_path,in_channel,out_channel)
        ir_image_list = dataset.list_images(args.test_ir_dir)
        vis_image_list = dataset.list_images(args.test_vis_dir)
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

            vutils.save_image(out_eval, os.path.join(out_image_path, f'{batch+1}.png'))
        print(total_time/10)
    print("Done......")





if __name__ == '__main__':
    main()


