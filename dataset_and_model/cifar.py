import os
import urllib
import numpy as np
import pickle
import torch
from typing import Tuple
from torch import nn
from torch.nn import functional as F
from torchvision import models


def load_batch(fpath):
    f = open(fpath, "rb").read()
    size = 32*32*3+1
    labels = []
    images = []
    for i in range(10000):
        arr = np.fromstring(f[i*size:(i+1)*size], dtype=np.uint8)
        lab = np.identity(10)[arr[0]]
        img = arr[1:].reshape((3, 32, 32))

        labels.append(lab)
        images.append((img/255)-.5)
    # return torch.tensor(images), torch.tensor(labels)
    return np.array(images), np.array(labels)


class CIFAR:
    def __init__(self, data_path: str,num_pic: int):
        train_data = []
        train_labels = []

        if not os.path.exists(data_path):
            urllib.request.urlretrieve("https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz",
                                       "../../../../../../../../dataset/cifar/cifar-data.tar.gz")
            os.popen("tar -xzf cifar-data.tar.gz").read()

        for i in range(5):
            r, s = load_batch(
                data_path + '/data_batch_'+str(i+1)+".bin")
            train_data.extend(r)
            train_labels.extend(s)

        train_data = np.array(train_data, dtype=np.float32)
        train_labels = np.array(train_labels)
        print('load cifar train data successfully')

        self.test_data, self.test_labels = load_batch(
            data_path + '/test_batch.bin')

        VALIDATION_SIZE = num_pic

        self.validation_data = train_data[:VALIDATION_SIZE, :, :, :]
        self.validation_labels = train_labels[:VALIDATION_SIZE]
        self.train_data = train_data[VALIDATION_SIZE:, :, :, :]
        self.train_labels = train_labels[VALIDATION_SIZE:]


class CIFARModelTorch(nn.Module):
    def __init__(self):
        super().__init__
        super(CIFARModelTorch, self).__init__()
        self.num_channels = 3
        self.image_size = 32
        self.num_labels = 10

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1,)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1,)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1,)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1,)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(3200, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        x = torch.reshape(x, (x.shape[0], -1))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.softmax(x)
        return x

def get_dataset_and_model(data_path: str, num_pics:int,  model_path: str, model_type) -> Tuple[np.ndarray, np.ndarray, torch.nn.Module]:
    cifar_dataset = CIFAR(data_path=data_path, num_pic=num_pics)
    if 'resnet' in model_type:
        model = load_resnet_model(model_type)
    else:
        model = CIFARModelTorch()
        model.load_state_dict(torch.load(model_path))
    model.eval() # set the module into evaluation mode

    images, labels = cifar_dataset.test_data[:num_pics, :, :, :], cifar_dataset.test_labels[:num_pics]


    # images, labels = cifar_dataset.test_data, cifar_dataset.test_labels
    # index_select_list = np.random.choice(range(10000),num_pics,replace=False)
    # images, labels = images[index_select_list], labels[index_select_list]
    return images, np.argmax(labels, axis=-1), model

model_dict={
    'resnet34': models.resnet34,
    'resnet18': models.resnet18,
}
def load_resnet_model(model_name):
    model = model_dict[model_name]
    model = models.resnet34(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.eval()
    return model