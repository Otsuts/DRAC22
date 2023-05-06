import torch
import os
import PIL
import torch.nn
from PIL import Image
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
# import moco.loader


class TrainData(torch.utils.data.Dataset):
    def __init__(self, root_dir, image_size=640, transform=True):
        # super(SBDataset, self).__init__()
        self.image_train_dir = os.path.join(root_dir, 'img/', 'all/')
        self.gts_path = os.path.join(root_dir, 'gt', 'labels.csv')
        self.load_imgs(self.image_train_dir)
        # print(self.train_img_paths)
        if not transform:
            self.transform = transforms.Compose(
                [transforms.Resize((image_size, image_size), interpolation=PIL.Image.BILINEAR),
                 transforms.CenterCrop((image_size, image_size)),
                 transforms.ToTensor(), ])
        else:
            self.transform = transforms.Compose(
                [transforms.RandomHorizontalFlip(),
                 transforms.RandomVerticalFlip(),
                #  transforms.RandomRotation(45),
                 transforms.Resize((image_size, image_size), interpolation=PIL.Image.BILINEAR),
                 transforms.CenterCrop((image_size, image_size)),
                 transforms.ToTensor()
                 ]
            )

        self.load_gts(self.gts_path)

    def __len__(self):
        return len(self.train_imgs)

    def __getitem__(self, idx):
        return self.transform(self.train_imgs[idx]), int(self.gts[idx])

    def load_imgs(self, dir):
        idx = 1
        self.train_imgs = []
        self.train_img_paths = []
        while idx < 2000:

            if os.path.exists(os.path.join(dir, f'{idx:03d}.png')):
                img_path = os.path.join(dir, f'{idx:03d}.png')
                # print(img_path)
                img = Image.open(img_path)
                img = img.convert('RGB')
                # print(img.mode)
                self.train_imgs.append(img)
                self.train_img_paths.append(f'{idx:03d}.png')
                # print(os.path.join(dir, f'{idx:03d}.png'))
            idx += 1

    def load_gts(self, path):
        with open(path, 'r') as file:
            data = file.read()

        data = data.split('\n')[1:]
        self.gts = []
        for idx, d in enumerate(data):
            d = d.split(',')
            name, value = d[0], int(d[1])
            self.gts.append(value)
            print(name, value)
            print(idx)
            if idx>=893:
                assert name == self.train_img_paths[idx][-len('0000.png'):]
            else:
                assert name == self.train_img_paths[idx][-len('000.png'):]


class TestData(torch.utils.data.Dataset):
    def __init__(self, root_dir, image_size=640):
        self.image_train_dir = os.path.join(root_dir, 'img/', 'test/')
        # self.gts_path = os.path.join(root_dir, 'gt', 'labels.csv')

        self.load_imgs(self.image_train_dir)
        self.transform = transforms.Compose(
            [transforms.Resize((image_size, image_size), interpolation=PIL.Image.BILINEAR),
             transforms.CenterCrop((image_size, image_size)),
             transforms.ToTensor(), ])
        # self.load_gts(self.gts_path)

    def __len__(self):
        return len(self.train_imgs)

    def __getitem__(self, idx):
        return self.transform(self.train_imgs[idx])  # , int(self.gts[idx])

    def load_imgs(self, dir):
        idx = 1
        self.train_imgs = []
        self.train_img_paths = []
        self.test_name = []
        while idx < 1500:

            if os.path.exists(os.path.join(dir, f'{idx:03d}.png')):
                img_path = os.path.join(dir, f'{idx:03d}.png')
                # print(img_path)
                img = Image.open(img_path)
                img = img.convert('RGB')
                # print(img.mode)
                self.train_imgs.append(img)
                self.test_name.append(f'{idx:03d}.png')
                self.train_img_paths.append(f'{idx:03d}.png')
                # print(os.path.join(dir, f'{idx:03d}.png'))
            idx += 1

    def names(self):
        return self.test_name
