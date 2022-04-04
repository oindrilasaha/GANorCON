from PIL import Image
import torchvision 
from torch.utils.data import Dataset

resnet_transform = torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
class ImageLabelDataset(Dataset):
    def __init__(
            self,
            img_path_list,
            img_size=(128, 128),
    ):
        self.img_path_list = img_path_list
        self.img_size = img_size

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, index):
        im_path = self.img_path_list[index]
        im = Image.open(im_path)
        im = self.transform(im)
        return im, im_path

    def transform(self, img):
        img = img.resize((self.img_size[0], self.img_size[1]))
        img = torchvision.transforms.ToTensor()(img)
        img = resnet_transform(img)
        return img