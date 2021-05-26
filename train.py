# -*- encoding: utf-8 -*-
"""
@File    : train.py
@Time    : 2021/3/29 10:39
@Author  : Chool
@Email   : depeng_cust@163.com
@Software: PyCharm
"""

import os, sys
from tqdm import trange
import torch
import torch.optim as optim
from torch.autograd import Variable
import dataset
from MIF_net import FusionNet
from args_fusion import args

from loss import msssim
from tensorboardX import SummaryWriter
from datetime import datetime
from torch.utils.data import DataLoader
import time
import argparse

if args.mult_device:
    device_ids = [0, 1]
else:
    device_ids = [0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# get lr
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def parse_args():
    parser = argparse.ArgumentParser(description="defeat detect")
    parser.add_argument("--echo", help="echo the string you use here")
    parser.add_argument("--epochs", default=10, help="train epochs", type=int)
    parser.add_argument("--resume", help="increase output verbosity",
                        action="store_true")
    parser.add_argument("--checkpoint", default="0", help="checkpoint num")
    return parser.parse_args()


def main():
    TIME_STAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
    train_log_dir = "./logs/tensorboard/train" + args.model_name + "_" + TIME_STAMP
    val_log_dir = "./logs/tensorboard/valid" + args.model_name + "_" + TIME_STAMP

    if os.path.exists(train_log_dir):
        os.makedirs(train_log_dir)
    if os.path.exists(val_log_dir):
        os.makedirs(val_log_dir)
    writer_train = SummaryWriter(log_dir=train_log_dir)
    writer_val = SummaryWriter(log_dir=val_log_dir)
    parse = parse_args()
    train(writer_train, writer_val, parse, train_state=None)


def eval(fusion_model, e, train_loss):
    torch.cuda.empty_cache()
    with torch.no_grad():
        fusion_model.eval()
        ir_image_list = dataset.list_images(args.eval_ir_dir)
        vis_image_list = dataset.list_images(args.eval_vis_dir)
        batch_size = args.batch_size_eval
        image_set_ir, batches = dataset.load_dataset(ir_image_list, batch_size)
        image_set_vis, _ = dataset.load_dataset(vis_image_list, batch_size)
        for batch in range(batches):
            image_paths_ir = image_set_ir[batch * batch_size:(batch * batch_size + batch_size)]
            image_paths_vis = image_set_vis[batch * batch_size:(batch * batch_size + batch_size)]
            img_ir = dataset.get_train_images(image_paths_ir)
            img_vis = dataset.get_train_images(image_paths_vis)

            if args.mult_device:
                img_ir = img_ir.cuda(device=device_ids[0])
                img_vis = img_vis.cuda(device=device_ids[0])
            else:
                img_ir = img_ir.cuda()
                img_vis = img_vis.cuda()

            out_eval = fusion_model(img_ir, img_vis)
            dataset.save_valid_image(out_eval, e, batch, train_loss)

    del img_ir, img_vis, out_eval
    torch.cuda.empty_cache()


def train(writer_train, writer_val, parse, train_state=None):
    in_c = 1
    input_nc = in_c
    output_nc = in_c

    with open(args.train_ir_list, "r") as F:
        ir_lines = F.readlines()
    with open(args.train_vis_list, "r") as F:
        vis_lines = F.readlines()
    with open(args.train_gt_list, "r") as F:
        gt_lines = F.readlines()

    ir_lines = ir_lines[:1000]
    vis_lines = vis_lines[:1000]
    gt_lines = gt_lines[:1000]

    print('BATCH SIZE %d.' % args.batch_size)
    print('Train images number %d.' % len(ir_lines))
    num_samples = len(ir_lines) // args.batch_size
    print('Train images samples %d.' % num_samples)

    fusion_dataset = dataset.Fusion_Dataset(ir_lines, vis_lines, gt_lines)
    fusion_dataloader = DataLoader(fusion_dataset, args.batch_size, shuffle=True)

    fusion_model = FusionNet(input_nc, output_nc)
    if args.mult_device:
        fusion_model = torch.nn.DataParallel(fusion_model, device_ids=device_ids)
        fusion_model = fusion_model.cuda(device=device_ids[0])
    else:
        fusion_model = fusion_model.cuda()

    optimizer = optim.Adam(fusion_model.parameters(), args.lr)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)  # 每个batch 调整一次学习率

    # loss
    l1_loss = torch.nn.L1Loss()
    ssim_loss = msssim

    print('Start training.....')
    tbar = trange(args.epochs)

    temp_path_model = os.path.join(args.save_model_dir, args.model_name)
    if os.path.exists(temp_path_model) is False:
        os.makedirs(temp_path_model)

    starting_epoch = 0
    if parse.resume:
        checkpoint_path = r'./models/checkpoint/' + parse.checkpoint
        train_state = torch.load(checkpoint_path)
    if train_state is not None:
        fusion_model.load_state_dict(train_state['model_state_dict'])
        optimizer.load_state_dict(train_state['optimizer_state_dict'])
        starting_epoch = train_state['epoch'] + 1

    best_loss = 100.
    for e in range(starting_epoch, args.epochs):
        print('\r Epoch %d.....' % (e + 1))

        all_ssim_loss_gt = 0.
        all_total_loss = 0.
        all_L1_loss = 0.
        for iteration, batch in enumerate(fusion_dataloader):
            img_ir, img_vis, img_gt = batch

            if args.mult_device:
                img_ir = img_ir.cuda(device=device_ids[0])
                img_vis = img_vis.cuda(device=device_ids[0])
                img_gt = img_gt.cuda(device=device_ids[0])
            else:
                img_ir = img_ir.cuda()
                img_vis = img_vis.cuda()
                img_gt = img_gt.cuda()

            img_ir = Variable(img_ir, requires_grad=False)
            img_vis = Variable(img_vis, requires_grad=False)
            img_gt = Variable(img_gt, requires_grad=False)

            optimizer.zero_grad()
            outputs = fusion_model(img_ir, img_vis)

            img_gt = Variable(img_gt.data.clone(), requires_grad=False)

            ssim_loss_gt = 1 - ssim_loss(outputs, img_gt, normalize=True)
            L1_loss = l1_loss(outputs, img_gt)

            ssim_loss_gt /= len(outputs)
            L1_loss /= len(outputs)

            # total loss
            total_loss = 84 * ssim_loss_gt + L1_loss * 16
            total_loss.backward()
            optimizer.step()

            all_ssim_loss_gt += ssim_loss_gt.item()
            all_total_loss += total_loss.item()
            all_L1_loss += L1_loss.item()

            if (iteration + 1) % args.log_interval == 0:

                mesg = "{} Epoch{}: [{}/{}] L1: {:.6f} ssim: {:.6f} total: {:.6f} lr:{:.6f}".format(
                    time.ctime(), (e + 1), iteration, num_samples,
                    all_L1_loss / args.log_interval,
                    all_ssim_loss_gt / args.log_interval,
                    all_total_loss / args.log_interval,
                    get_lr(optimizer)
                )
                tbar.set_description(mesg)

                writer_train.add_scalar("total_loss", all_total_loss / args.log_interval, iteration + (e))
                writer_train.add_scalar("SSIM_loss", all_ssim_loss_gt / args.log_interval, iteration + (e))
                writer_train.add_scalar("L1_loss", all_L1_loss / args.log_interval, iteration + (e))

                if (all_total_loss / args.log_interval < best_loss) and e != 0:
                    best_loss = all_total_loss / args.log_interval

                    checkpoint = {
                        'epoch': e,
                        'model_state_dict': fusion_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }
                    save_model_filename = temp_path_model + "/" + args.model_name + "epoch_" + str(e + 1) + "_" + \
                                          "loss_" + str(round(best_loss, 6)) + \
                                          str(time.ctime()).replace(" ", "_").replace(":", "_") + "_plk"
                    torch.save(checkpoint, save_model_filename)
                    print("\n Validation...")
                    time.sleep(0.5)
                    del img_ir, img_vis, img_gt, outputs, \
                        all_ssim_loss_gt, all_total_loss, all_L1_loss, total_loss

                    eval(fusion_model, e, best_loss)
                    fusion_model.train()
                all_ssim_loss_gt = 0.
                all_total_loss = 0.
                all_L1_loss = 0.

        lr_scheduler.step()  # update learning rate
    writer_val.close()
    writer_train.close()
    print("\nDone, trained model saved at {}".format(temp_path_model))


if __name__ == "__main__":
    main()
