import os
import urllib.request
import numpy as np
import gzip
import torch
from torch import nn
from typing import Tuple
from torch.nn import functional as F

def extract_data(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(num_images*28*28)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = (data / 255) - 0.5
        data = data.reshape(num_images, 1, 28, 28)
        return data

def extract_labels(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8)
    return (np.arange(10) == labels[:, None]).astype(np.float32)

class MNIST:
    def __init__(self, data_path: str, num_pic: int):
        if not os.path.exists(data_path):
            os.mkdir(data_path)
            files = ["train-images-idx3-ubyte.gz",
                     "t10k-images-idx3-ubyte.gz",
                     "train-labels-idx1-ubyte.gz",
                     "t10k-labels-idx1-ubyte.gz"]
            for name in files:

                urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/' + name, "data/"+name)

        train_data = extract_data(data_path + "/train-images-idx3-ubyte.gz", 60000)
        train_labels = extract_labels(data_path + "/train-labels-idx1-ubyte.gz", 60000)
        self.test_data = extract_data(data_path + "/t10k-images-idx3-ubyte.gz", 10000)
        self.test_labels = extract_labels(data_path + "/t10k-labels-idx1-ubyte.gz", 10000)
        
        VALIDATION_SIZE = num_pic
        
        self.validation_data = train_data[:VALIDATION_SIZE, :, :, :]
        self.validation_labels = train_labels[:VALIDATION_SIZE]
        self.train_data = train_data[VALIDATION_SIZE:, :, :, :]
        self.train_labels = train_labels[VALIDATION_SIZE:]


class MNISTModelTorch(nn.Module):
    def __init__(self, channels=1, image_size=28, num_labels=10) -> None:
        # super().__init__
        super(MNISTModelTorch, self).__init__()
        self.num_channels = channels
        self.image_size = image_size
        self.num_labels = num_labels

        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3,3), stride=(1,1),)
        self.ac1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=(1,1),)
        self.ac2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3,3), stride=(1,1),)
        self.ac3 = nn.ReLU()
        self.conv4 = nn.Conv2d(64, 64, kernel_size=(3,3), stride=(1,1),)
        self.ac4 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(1024, 200)
        self.ac5 = nn.ReLU()
        self.fc2 = nn.Linear(200, 200)
        self.ac6 = nn.ReLU()
        self.fc3 = nn.Linear(200, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.ac1(x)
        x = self.conv2(x)
        x = self.ac2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.ac3(x)
        x = self.conv4(x)
        x = self.ac4(x)
        x = self.pool2(x)
        x = x.view(x.shape[0], -1)
        # x = x.flatten(1, -1)
        x = self.fc1(x)
        x = self.ac5(x)
        x = self.fc2(x)
        x = self.ac6(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x

def get_dataset_and_model(data_path: str, num_pics:int,  model_path: str, model_type) -> Tuple[np.ndarray, np.ndarray, torch.nn.Module]:
    np.random.seed
    mnist_dataset = MNIST(data_path=data_path, num_pic=num_pics)
    model = MNISTModelTorch()
    model.load_state_dict(torch.load(model_path))
    model.eval() # set the module into evaluation mode
    # images, labels = mnist_dataset.test_data, mnist_dataset.test_labels
    # index_select_list = np.random.choice(range(10000),num_pics,replace=False)
    # images, labels = images[index_select_list], labels[index_select_list]
    images, labels = mnist_dataset.test_data[:num_pics, :, :, :], mnist_dataset.test_labels[:num_pics]
    return images, np.argmax(labels, axis=-1), model        
            