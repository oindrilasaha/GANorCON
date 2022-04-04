import torch
import torchvision.datasets as dsets
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
import os
import numpy as np
import random

class CelebAMaskHQ():
    def __init__(self, img_path, label_path, transform_img, transform_label, mode):
        self.img_path = img_path
        self.label_path = label_path
        self.transform_img = transform_img
        self.transform_label = transform_label
        self.train_dataset = []
        self.test_dataset = []
        self.mode = mode
        self.preprocess_nvidia()
        
        if mode == True:
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def preprocess_nvidia(self):
        if self.mode==True:
            for i in range(int(len([name for name in os.listdir(self.img_path) if os.path.isfile(os.path.join(self.img_path, name))])/2)):
                img_path = os.path.join(self.img_path, 'image_'+str(i)+'.jpg')
                label_path = os.path.join(self.label_path, 'image_mask'+str(i)+'.npy')
                self.train_dataset.append([img_path, label_path])
        else:
            for i in range(int(len([name for name in os.listdir(self.img_path) if os.path.isfile(os.path.join(self.img_path, name))])/2)):
                img_path = os.path.join(self.img_path, 'face_'+str(i)+'.png')
                label_path = os.path.join(self.label_path, 'mask_'+str(i)+'.npy')
                self.test_dataset.append([img_path, label_path])

        print('Finished preprocessing the Nvidia dataset...')

    def __getitem__(self, index):   

        dataset = self.train_dataset if self.mode == True else self.test_dataset
        img_path, label_path = dataset[index]
        image = Image.open(img_path)
        label = np.load(label_path)

        label = Image.fromarray(label)
        image = image.resize((512,512))
        label = label.resize((512, 512), resample=Image.NEAREST)

        crop = random.random() < 0.5
        if crop and self.mode==True:
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                image, scale=(0.6,1.0), ratio=(0.7,1.3))

            image = TF.crop(image, i, j, h, w)
            label = TF.crop(label, i, j, h, w)

            image = image.resize((512,512))
            label = label.resize((512, 512), resample=Image.NEAREST)

        jitter = random.random() < 0.5
        if jitter and self.mode==True:
            image = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0)(image)

        hflip = random.random() < 0.5
        if hflip and self.mode==True:
          image = image.transpose(Image.FLIP_LEFT_RIGHT)
          label = label.transpose(Image.FLIP_LEFT_RIGHT)
        label = np.array(label, dtype=np.long)

        return self.transform_img(image), self.transform_label(label)

    def __len__(self):
        """Return the number of images."""
        return self.num_images

class Data_Loader():
    def __init__(self, img_path, label_path, image_size, batch_size, mode):
        self.img_path = img_path
        self.label_path = label_path
        self.imsize = image_size
        self.batch = batch_size
        self.mode = mode

    def transform_img(self, resize, totensor, normalize, centercrop):
        options = []
        if centercrop:
            options.append(transforms.CenterCrop(160))
        if resize:
            options.append(transforms.Resize((self.imsize,self.imsize)))
        if totensor:
            options.append(transforms.ToTensor())
        if normalize:
            options.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]))
        transform = transforms.Compose(options)
        return transform

    def transform_label(self, resize, totensor, normalize, centercrop):
        options = []
        if centercrop:
            options.append(transforms.CenterCrop(160))
        if resize:
            options.append(transforms.Resize((self.imsize,self.imsize)))
        if totensor:
            options.append(transforms.ToTensor())
        if normalize:
            options.append(transforms.Normalize((0, 0, 0), (0, 0, 0)))
        transform = transforms.Compose(options)
        return transform

    def loader(self):
        transform_img = self.transform_img(True, True, True, False) 
        transform_label = self.transform_label(False, True, False, False)  
        dataset = CelebAMaskHQ(self.img_path, self.label_path, transform_img, transform_label, self.mode)

        print(len(dataset))

        loader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=self.batch,
                                             shuffle=False,#self.mode==True,
                                             num_workers=0,
                                             drop_last=False,
                                             pin_memory=True)
        return loader
