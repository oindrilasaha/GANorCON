from __future__ import print_function

import os
import sys
import time
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import argparse
import socket
import torch.multiprocessing as mp
import torch.distributed as dist

import tensorboard_logger as tb_logger

from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader

from utils.util import adjust_learning_rate, AverageMeter, Tee

from models.resnet import InsResNet50,InsResNet18,InsResNet34,InsResNet101,InsResNet152
from models.segmentor import fcn, UNet
from models.loss import cross_entropy2d

from data_loader.data_loader_celebamask import Data_Loader
from data_loader.data_loader_forgen import ImageLabelDataset

import matplotlib.pyplot as plt

import numpy as np
import random
import math
import cv2

torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



def parse_option():

    hostname = socket.gethostname()

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=20, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=2, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=60, help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.1, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='30,40,50', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2, help='decay rate for learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam')

    # model definition
    parser.add_argument('--model', type=str, default='resnet50', 
                        choices=['resnet50', 'resnet50x2', 'resnet50x4', 'hourglass',
                                 'resnet18', 'resnet34', 'resnet101', 'resnet152'])
    parser.add_argument('--segmodel', type=str, default='fcn', 
                        choices=['fcn', 'UNet'])

    parser.add_argument('--trained_model_path', type=str, default=None, help='pretrained backbone')
    parser.add_argument('--layer', type=int, default=3, help='resnet layers')


    # model path and name  
    parser.add_argument('--model_name', type=str, default="face_model") # moco_version, network, input_size, crop_size
    parser.add_argument('--model_path', type=str, default="./512_faces_celeba") # path to store the models

    # resume
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    parser.add_argument('--image_size', type=int, default=512, help='image size') # image preprocessing
    parser.add_argument('--generate', action='store_true', help='generate dataset for deeplab')
    parser.add_argument('--gen_path', type=str, default=None)

    # add BN
    parser.add_argument('--bn', action='store_true', help='use parameter-free BN')
    parser.add_argument('--cosine', action='store_true', help='use cosine annealing')
    parser.add_argument('--multistep', action='store_true', help='use multistep LR')
    parser.add_argument('--adam', action='store_true', help='use adam optimizer')
    parser.add_argument('--amsgrad', action='store_true', help='use amsgrad for adam')


    # GPU setting
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')

    # log_path
    parser.add_argument('--log_path', default='log_tmp', type=str, metavar='PATH', help='path to the log file')

    # use hypercolumn or single layer output
    parser.add_argument('--use_hypercol', action='store_true', help='use hypercolumn as representations')

    opt = parser.parse_args()




    opt.save_path = opt.model_path
    opt.tb_path = '%s_tensorboard' % opt.model_path

    Tee(opt.log_path, 'a')

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.save_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def main():

    global best_error
    best_error = np.Inf

    args = parse_option()

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    train_loader_fn = Data_Loader(img_path='./DatasetGAN_data/annotation/training_data/face_processed/',
                                         label_path='./DatasetGAN_data/annotation/training_data/face_processed/',
                                         image_size=args.image_size, 
                                         batch_size=args.batch_size,
                                         mode=True)
    val_loader_fn = Data_Loader(img_path='./DatasetGAN_data/annotation/testing_data/face_34_class/',
                                         label_path='./DatasetGAN_data/annotation/testing_data/face_34_class/',
                                         image_size=args.image_size, 
                                         batch_size=args.batch_size,
                                         mode=False)


    train_sampler = None

    train_loader = train_loader_fn.loader()
    val_loader = val_loader_fn.loader()

    # create model and optimizer
    input_size = args.image_size 
    pool_size = int(input_size / 2**5) 

    if args.model == 'resnet50':
        model = InsResNet50(pool_size=pool_size)#, pretrained=True)
        desc_dim = {1:64, 2:256, 3:512, 4:1024, 5:2048}
    elif args.model == 'resnet50x2':
        model = InsResNet50(width=2, pool_size=pool_size)
        desc_dim = {1:128, 2:512, 3:1024, 4:2048, 5:4096}
    elif args.model == 'resnet50x4':
        model = InsResNet50(width=4, pool_size=pool_size)
        desc_dim = {1:512, 2:1024, 3:2048, 4:4096, 5:8192}
    elif args.model == 'resnet18':
        model = InsResNet18(width=1, pool_size=pool_size)
        desc_dim = {1:64, 2:64, 3:128, 4:256, 5:512}
    elif args.model == 'resnet34':
        model = InsResNet34(width=1, pool_size=pool_size)
        desc_dim = {1:64, 2:64, 3:128, 4:256, 5:512}
    elif args.model == 'resnet101':
        model = InsResNet101(width=1, pool_size=pool_size)
        desc_dim = {1:64, 2:256, 3:512, 4:1024, 5:2048}
    elif args.model == 'resnet152':
        model = InsResNet152(width=1, pool_size=pool_size)
        desc_dim = {1:64, 2:256, 3:512, 4:1024, 5:2048}
    elif args.model == 'hourglass':
        model = HourglassNet()
    else:
        raise NotImplementedError('model not supported {}'.format(args.model))


    if args.model == 'hourglass':
        feat_dim = 64
    else:
        if args.use_hypercol:
            feat_dim = 0
            for i in range(args.layer):
                feat_dim += desc_dim[5-i]
        else:
            feat_dim = desc_dim[args.layer]

    if args.segmodel=='fcn':
        segmentor = fcn(feat_dim, n_classes=34)
    else:
        segmentor = UNet(feat_dim, n_classes=34)

    
    print('==> loading pre-trained model')
    ckpt = torch.load(args.trained_model_path, map_location='cpu')
    state_dict = ckpt['model']

    for key in list(state_dict.keys()):
        state_dict[key.replace('module.encoder', 'encoder.module')] = state_dict.pop(key)

    model.load_state_dict(state_dict, strict=False)
    print('==> done')

    segmentor.init_weights()

    model = model.cuda()
    segmentor = segmentor.cuda()

    if args.generate==True:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=False)            
        segmentor.load_state_dict(checkpoint['segmentor'])

        images_togen = []
        img_path_base = './CelebAMask-HQ/train_img/'

        for i in range(len([name for name in os.listdir(img_path_base) if os.path.isfile(os.path.join(img_path_base, name))])):
            img_path = os.path.join(img_path_base, str(i)+'.jpg')
            images_togen.append(img_path)
            if i==10000:
                break
        gen_data = ImageLabelDataset(img_path_list=images_togen,
                            img_size=(args.image_size, args.image_size))
        if not os.path.isdir(args.gen_path):
            os.mkdir(args.gen_path)
        model.eval()
        segmentor.eval()  
        gen_data = DataLoader(gen_data, batch_size=1, shuffle=False, num_workers=16)
        with torch.no_grad():
            for idx, (input, im_path) in enumerate(gen_data):                              
                input = input.cuda()
                input = input.float()
                # compute output
                feat = model(input, args.layer, args.use_hypercol, (512,512))
                feat = feat.detach()
                output = segmentor(feat)
                output = output.detach()
                label_out = torch.nn.functional.log_softmax(output,dim=1)
                label_out = label_out.view(1, 34, 512, 512)
                label = label_out[0]
                label = label.data.max(0)[1].cpu().numpy()
                cv2.imwrite(os.path.join(args.gen_path, str(idx) +'.png'), label)
                if idx%100==0:
                    print('Processed '+str(idx)+'/'+str(10000)) 
        return

    criterion = cross_entropy2d

    if not args.adam:
        optimizer = torch.optim.SGD(segmentor.parameters(),
                                    lr=args.learning_rate,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(segmentor.parameters(),
                                     lr=args.learning_rate,
                                     betas=(args.beta1, args.beta2),
                                     weight_decay=args.weight_decay,
                                     eps=1e-8,
                                     amsgrad=args.amsgrad)
    model.eval()
    cudnn.benchmark = True

    # optionally resume from a checkpoint
    args.start_epoch = 1
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            # checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch'] + 1
            segmentor.load_state_dict(checkpoint['segmentor'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_error = checkpoint['best_error']
            # best_error = best_error.cuda()
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            if 'opt' in checkpoint.keys():
                # resume optimization hyper-parameters
                print('=> resume hyper parameters')
                if 'bn' in vars(checkpoint['opt']):
                    print('using bn: ', checkpoint['opt'].bn)
                if 'adam' in vars(checkpoint['opt']):
                    print('using adam: ', checkpoint['opt'].adam)
                if 'cosine' in vars(checkpoint['opt']):
                    print('using cosine: ', checkpoint['opt'].cosine)
                args.learning_rate = checkpoint['opt'].learning_rate
                # args.lr_decay_epochs = checkpoint['opt'].lr_decay_epochs
                args.lr_decay_rate = checkpoint['opt'].lr_decay_rate
                args.momentum = checkpoint['opt'].momentum
                args.weight_decay = checkpoint['opt'].weight_decay
                args.beta1 = checkpoint['opt'].beta1
                args.beta2 = checkpoint['opt'].beta2
            del checkpoint
            torch.cuda.empty_cache()
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # set cosine annealing scheduler
    if args.cosine:

        eta_min = args.learning_rate * (args.lr_decay_rate ** 3) * 0.1
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min, -1)
        # dummy loop to catch up with current epoch
        for i in range(1, args.start_epoch):
            scheduler.step()
    elif args.multistep:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 250], gamma=0.1)
        # dummy loop to catch up with current epoch
        for i in range(1, args.start_epoch):
            scheduler.step()

    # tensorboard
    logger = tb_logger.Logger(logdir=args.tb_folder, flush_secs=2)
    train_loss_list = []
    test_loss_list = []


    # routine
    for epoch in range(args.start_epoch, args.epochs + 1):

        if args.cosine or args.multistep:
            scheduler.step()
        else:
            adjust_learning_rate(epoch, args, optimizer)
        print("==> training...")

        time1 = time.time()
        train_loss = train(epoch, train_loader, model, segmentor, criterion, optimizer, args)
        time2 = time.time()
        print('train epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # logger.log_value('InterOcularError', InterOcularError, epoch)
        train_loss_list.append(train_loss)
        logger.log_value('train_loss', train_loss, epoch)
        logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        print("==> testing...")
        test_loss = validate(val_loader, model, segmentor, criterion, args)

        test_loss_list.append(test_loss)

        # logger.log_value('Test_InterOcularError', test_InterOcularError, epoch)
        logger.log_value('test_loss', test_loss, epoch) 

        # save the best model
        if test_loss < best_error:
            best_error = test_loss
            state = {
                'opt': args,
                'epoch': epoch,
                'model': model.state_dict(),
                'segmentor': segmentor.state_dict(),
                'best_error': best_error,
                'optimizer': optimizer.state_dict(),
            }
            save_name = '{}.pth'.format(args.model)
            save_name = os.path.join(args.save_folder, save_name)
            print('saving best model!')
            torch.save(state, save_name)

        # save model
        if epoch % args.save_freq == 0:
            print('==> Saving...')
            state = {
                'opt': args,
                'epoch': epoch,
                'segmentor': segmentor.state_dict(),
                'best_error': test_loss,
                'optimizer': optimizer.state_dict(),
            }
            save_name = 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch)
            save_name = os.path.join(args.save_folder, save_name)
            print('saving regular model!')
            torch.save(state, save_name)

        # tensorboard logger
        pass

    x=range(len(train_loss_list))

    plt.plot(x, train_loss_list, label = "train loss")
    plt.plot(x, test_loss_list, label = "test loss")
    plt.xlabel('epochs')
    plt.ylabel('loss')

    plt.savefig(os.path.join(args.save_folder,'loss_curve.png'))


def set_lr(optimizer, lr):
    """
    set the learning rate
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(epoch, train_loader, model, segmentor, criterion, optimizer, opt):
    """
    one epoch training
    """

    model.eval()
    segmentor.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    # InterOcularError = AverageMeter()

    end = time.time()
    for idx, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda(opt.gpu, non_blocking=True)
        input = input.float()
        target = target.cuda(opt.gpu, non_blocking=True)

        # ===================forward=====================
        with torch.no_grad():
            feat = model(input, opt.layer, opt.use_hypercol, (512,512))
            feat = feat.detach()

        output = segmentor(feat)
        loss = criterion(output, target)

        if idx == 0:
            print('Layer:{0}, shape of input:{1}, feat:{2}, output:{3}'.format(opt.layer, 
                                input.size(), feat.size(), output.size()))

        losses.update(loss.item(), input.size(0))

        # ===================backward=====================
        
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   epoch, idx, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))#, InterOcularError=InterOcularError))
            sys.stdout.flush()

    return losses.avg


def validate(val_loader, model, segmentor, criterion, opt):
    batch_time = AverageMeter()
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()
    segmentor.eval()

    with torch.no_grad():
        end = time.time()
        for idx, (input, target) in enumerate(val_loader):
            if opt.gpu is not None:
                input = input.cuda(opt.gpu, non_blocking=True)
            input = input.float()
            target = target.cuda(opt.gpu, non_blocking=True)

            # compute output
            feat = model(input, opt.layer, opt.use_hypercol, (512,512))
            feat = feat.detach()

            output = segmentor(feat)
            loss = criterion(output, target)

            losses.update(loss.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    idx, len(val_loader), batch_time=batch_time, loss=losses))

    return losses.avg


if __name__ == '__main__':
    best_error = np.Inf
    main()
