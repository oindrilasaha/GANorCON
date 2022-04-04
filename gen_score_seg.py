import sys
sys.path.append('../')
import os
import argparse
import gc
import os
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import glob
import json
import numpy as np
from data_loader.data_loader_celebamask import Data_Loader
from models.resnet import InsResNet50,InsResNet18,InsResNet34,InsResNet101,InsResNet152
from models.segmentor import fcn, UNet
from torchvision import transforms

import torch.nn.functional as F
import cv2


from PIL import Image

import scipy.misc


def process_image(images):
    drange = [-1, 1]
    scale = 255 / (drange[1] - drange[0])
    images = images * scale + (0.5 - drange[0] * scale)

    images = images.astype(int)
    images[images > 255] = 255
    images[images < 0] = 0

    return images.astype(int)

class ImageLabelDataset(Dataset):
    def __init__(
            self,
            img_path_list,
            label_path_list,
            img_size=(128, 128),
    ):
        self.img_path_list = img_path_list
        self.label_path_list = label_path_list
        self.img_size = img_size

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, index):
        im_path = self.img_path_list[index]
        lbl_path = self.label_path_list[index]
        im = Image.open(im_path)
        try:
            lbl = np.load(lbl_path)
        except:
            lbl = np.array(Image.open(lbl_path))
        if len(lbl.shape) == 3:
            lbl = lbl[:, :, 0]

        lbl = Image.fromarray(lbl.astype('uint8'))
        im, lbl = self.transform(im, lbl)

        return im, lbl, im_path

    def transform(self, img, lbl):
        img = img.resize((512, 512))
        lbl = lbl.resize((self.img_size[0], self.img_size[1]), resample=Image.NEAREST)
        lbl = torch.from_numpy(np.array(lbl)).long()
        img = transforms.ToTensor()(img)
        return img, lbl


def cross_validate(args):
    
    ignore_index = -1
    cp_path = args.resume
    base_path = os.path.join(cp_path, "cross_validation")
    if not os.path.exists(base_path):
        os.mkdir(base_path)

    cps_all = glob.glob(cp_path + "/*")
    cp_list = [data for data in cps_all if '.pth' in data and 'resnet' not in data]
    cp_list.sort()

    ids = range(34)

    data_all = glob.glob('./DatasetGAN_data/annotation/testing_data/face_34_class/' + "/*")
    images = [path for path in data_all if 'npy' not in path]
    labels = [path for path in data_all if 'npy' in path]
    images.sort()
    labels.sort()


    fold_num =int( len(images) / 5)
    resnet_transform = torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])

    input_size = 512 
    pool_size = int(input_size / 2**5)
    model = InsResNet50(pool_size=pool_size, pretrained=True)
    desc_dim = {1:64, 2:256, 3:512, 4:1024, 5:2048}

    feat_dim = 0
    for i in range(4):
        feat_dim += desc_dim[5-i]

    if args.distill == True:
        segmentor = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=False, progress=False,                                                                      num_classes=34, aux_loss=None)
    elif args.model == "UNet":
        segmentor = UNet(feat_dim, n_classes=34)
    elif args.model == "fcn":
        segmentor = fcn(feat_dim, n_classes=34)

    cross_mIOU = []

    for i in range(5):
        val_image = images[fold_num * i: fold_num *i + fold_num]
        val_label = labels[fold_num * i: fold_num *i + fold_num]
        test_image = [img for img in images if img not in val_image]
        test_label =[label for label in labels if label not in val_label]
        print("Val Data length,", str(len(val_image)))
        print("Testing Data length,", str(len(test_image)))

        val_data = ImageLabelDataset(img_path_list=val_image,
                                      label_path_list=val_label,
                                      img_size=(512, 512))
        val_data = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=0)

        test_data = ImageLabelDataset(img_path_list=test_image,
                                  label_path_list=test_label,
                                img_size=(512, 512))
        test_data = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)

        best_miou = 0
        best_val_miou = 0

        for i_ckpt, ckpt_name in enumerate(cp_list):
                if i_ckpt==0 and args.distill==False:
                    ckpt = torch.load(os.path.join(cp_path, 'resnet50.pth'))
                    state_dict = ckpt['model']
                    model.load_state_dict(state_dict, strict=False)
                # import pdb;pdb.set_trace()
                print(args)
                if args.distill==False:
                    model.cuda()
                    model.eval()
                    ckpt_seg = torch.load(ckpt_name)
                    segmentor.load_state_dict(ckpt_seg['segmentor'])
                else:
                    ckpt_seg = torch.load(ckpt_name)
                    segmentor.load_state_dict(ckpt_seg['model_state_dict'])

                segmentor.cuda()
                segmentor.eval()

                unions = {}
                intersections = {}
                for target_num in ids:
                    unions[target_num] = 0
                    intersections[target_num] = 0

                with torch.no_grad():
                    for idxx, da, in enumerate(val_data):

                        img, mask = da[0], da[1]

                        if img.size(1) == 4:
                            img = img[:, :-1, :, :]

                        img = img.cuda()
                        mask = mask.cuda()
                        input_img_tensor = []
                        for b in range(img.size(0)):
                            input_img_tensor.append(resnet_transform(img[b]))
                        input_img_tensor = torch.stack(input_img_tensor)

                        if args.distill == False:
                            feat = model(input_img_tensor, 4, True, (512,512))
                            feat = feat.detach()
                            y_pred = segmentor(feat)
                        else:
                            y_pred = segmentor(input_img_tensor)['out']

                        y_pred = torch.log_softmax(y_pred, dim=1)
                        _, y_pred = torch.max(y_pred, dim=1)
                        y_pred = y_pred.cpu().detach().numpy()

                        mask = mask.cpu().detach().numpy()
                        bs = y_pred.shape[0]

                        curr_iou = []
                        if ignore_index > 0:
                            y_pred = y_pred * (mask != ignore_index)
                        for target_num in ids:
                            y_pred_tmp = (y_pred == target_num).astype(int)
                            mask_tmp = (mask == target_num).astype(int)

                            intersection = (y_pred_tmp & mask_tmp).sum()
                            union = (y_pred_tmp | mask_tmp).sum()

                            unions[target_num] += union
                            intersections[target_num] += intersection

                            if not union == 0:
                                curr_iou.append(intersection / union)
                    mean_ious = []

                    for target_num in ids:
                        mean_ious.append(intersections[target_num] / (1e-8 + unions[target_num]))
                    mean_iou_val = np.array(mean_ious).mean()

                    if mean_iou_val > best_val_miou:
                        best_val_miou = mean_iou_val
                        unions = {}
                        intersections = {}
                        for target_num in ids:
                            unions[target_num] = 0
                            intersections[target_num] = 0

                        with torch.no_grad():
                            testing_vis = []
                            for idxx, da, in enumerate(test_data):

                                img, mask = da[0], da[1]

                                if img.size(1) == 4:
                                    img = img[:, :-1, :, :]

                                img = img.cuda()
                                mask = mask.cuda()
                                input_img_tensor = []
                                for b in range(img.size(0)):
                                    input_img_tensor.append(resnet_transform(img[b]))
                                input_img_tensor = torch.stack(input_img_tensor)

                                if args.distill == False:
                                    feat = model(input_img_tensor, 4, True, (512,512))
                                    feat = feat.detach()
                                    y_pred = segmentor(feat)
                                else:
                                    y_pred = segmentor(input_img_tensor)['out']

                                y_pred = torch.log_softmax(y_pred, dim=1)
                                _, y_pred = torch.max(y_pred, dim=1)
                                y_pred = y_pred.cpu().detach().numpy()


                                mask = mask.cpu().detach().numpy()

                                curr_iou = []
                                if ignore_index > 0:
                                    y_pred = y_pred * (mask != ignore_index)
                                for target_num in ids:
                                    y_pred_tmp = (y_pred == target_num).astype(int)
                                    mask_tmp = (mask == target_num).astype(int)

                                    intersection = (y_pred_tmp & mask_tmp).sum()
                                    union = (y_pred_tmp | mask_tmp).sum()

                                    unions[target_num] += union
                                    intersections[target_num] += intersection

                                    if not union == 0:
                                        curr_iou.append(intersection / union)


                                img = img.cpu().numpy()
                                img =  img * 255.
                                img = np.transpose(img, (0, 2, 3, 1)).astype(np.uint8)

                            test_mean_ious = []

                            for target_num in ids:
                                test_mean_ious.append(intersections[target_num] / (1e-8 + unions[target_num]))

                            best_test_miou = np.array(test_mean_ious).mean()


                            print("Best IOU ,", str(best_test_miou), "CP: ", ckpt_name)

        cross_mIOU.append(best_test_miou)

    print(cross_mIOU)
    print(" cross validation mean:" , np.mean(cross_mIOU) )
    print(" cross validation std:", np.std(cross_mIOU))
    result = {"Cross validation mean": np.mean(cross_mIOU), "Cross validation std": np.std(cross_mIOU), "Cross validation":cross_mIOU }



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str,  default="")
    parser.add_argument('--distill', action='store_true')
    parser.add_argument('--model', type=str,  default="UNet")

    args = parser.parse_args()
    # import pdb;pdb.set_trace()
    print(args)

    cross_validate(args)
