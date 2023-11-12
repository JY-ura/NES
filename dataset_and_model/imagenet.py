import PIL
import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple
from torchvision import datasets, transforms
import torchvision

def load_image(path, image_size):
    image = PIL.Image.open(path)
    if image.height > image.width:
        height_off = int((image.height - image.width)/2)
        image = image.crop(
            (0, height_off, image.width, height_off+image.width))
    elif image.width > image.height:
        width_off = int((image.width - image.height)/2)
        image = image.crop(
            (width_off, 0, width_off+image.height, image.height))
    image = image.resize((image_size, image_size))
    img = np.asarray(image).astype(np.float32) / 255 - 0.5
    if img.ndim == 2:
        img = np.repeat(img[:, :, np.newaxis], repeats=3, axis=2)
    if img.shape[2] == 4:
        # alpha channel
        img = img[:, :, :3]
    return img.transpose(2, 0, 1)


# def get_image(index, imagenet_path=None):
#     data_path = os.path.join(imagenet_path, 'val')
#     image_paths = sorted([os.path.join(data_path, i)
#                          for i in os.listdir(data_path)])
#     # assert len(image_paths) == 50000
#     labels_path = os.path.join(imagenet_path, 'val.txt')
#     with open(labels_path) as labels_file:
#         labels = [i.split(' ') for i in labels_file.read().strip().split('\n')]
#         labels = {os.path.basename(i[0]): int(i[1]) for i in labels}

#     def get(index):
#         path = image_paths[index]
#         x = load_image(path)
#         y = labels[os.path.basename(path)]
#         return x, y
#     return get(index)


class ImageNet(Dataset):
    def __init__(self, imagenet_path: str, image_size) -> None:
        super().__init__()
        # self.test_datas = np.load(imagenet_path+"imagenet_test_data.npy")[30:40]
        # self.test_labels = np.load(imagenet_path+ "imagenet_test_labels.npy")[30:40]
        data_path = os.path.join(imagenet_path, 'val')
        self.image_paths = sorted([os.path.join(data_path, i)
                                   for i in os.listdir(data_path)])
        self.image_size = image_size
        # assert len(self.image_paths) < 100
        self.len = len(self.image_paths)
        labels_path = os.path.join(imagenet_path, 'val.txt')
        with open(labels_path) as labels_file:
            labels = [i.split(' ')
                      for i in labels_file.read().strip().split('\n')]
            self.labels = {os.path.basename(i[0]): int(i[1]) for i in labels}

    def __getitem__(self, index: int):
        path = self.image_paths[index]
        img = load_image(path, self.image_size)
        label = self.labels[os.path.basename(path)]
        return img, label

    def __len__(self,):
        return self.len


def get_imgnet_images(imgagenet_path, num_pics, image_size):
    dataset = ImageNet(imgagenet_path, image_size)
    loader = DataLoader(dataset, batch_size=num_pics,
                        shuffle=False)
    images, labels = next(iter(loader))
    # images, labels = dataset.test_datas, dataset.test_labels
    # images = images.transpose(0,3,1,2)
    # print(np.max(images[0])
    # print(np.min(images[0])
    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int64)


def get_dataset_and_model(data_path: str, num_pics: int, model_path, model_type: str) -> Tuple[np.ndarray, np.ndarray, torch.nn.Module]:
    
    if model_type == 'inceptionv3':
        model = torch.hub.load('pytorch/vision:v0.10.0',
                            'inception_v3', pretrained=True)
        image_size = 299
    elif model_type == 'VIT':
        model = torchvision.models.vision_transformer.vit_b_16(
            weights=torchvision.models.vision_transformer.ViT_B_16_Weights.IMAGENET1K_V1
        )
        image_size = 224
    else:
        print("model load error!")
    # print("model: ", model_type)
    model.eval()  # set the module into evaluation mode
    images, labels = get_imgnet_images(
        imgagenet_path=data_path, num_pics=num_pics, image_size=image_size)
    return images, labels, model