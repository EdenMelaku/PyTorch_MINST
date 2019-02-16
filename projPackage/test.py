import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import scipy.ndimage


import cv2

#this are the hyper parametrs that we are going to use to train owr model
from PIL import Image
from dask.array.tests.test_stats import scipy

n_epochs = 3
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10


random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)



class Net(nn.Module):
    print("defining neural netwok")

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)




def Example():
  print("running examples")
  with torch.no_grad():
      example_data=convert("/home/eden/Pictures/index.png")
      #data = np.vectorize(lambda x: 255 - x)(np.ndarray.flatten(scipy.ndimage.imread("/home/eden/Pictures/index.png", flatten=True)))
      da=torch.from_numpy(example_data)
      print(da)
      output = network(da)
      print("prediction:{}".format(
      output.data.max(1, keepdim=True)))

import numpy as np
def convertMinst(path):

    img = Image.open(path).convert("L")
    img = np.resize(img, (28, 28, 1))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1, 28, 28, 1)/255
    print(im2arr)
    return im2arr


def convert(path):
    # create an array where we can store our 4 pictures
    images = np.zeros((1, 784))
    # and the correct values
    correct_vals = np.zeros((1, 10))

    # we want to test our images which you saw at the top of this page

    # read the image
    gray = cv2.imread(path, cv2.COLOR_BGR2GRAY)

    # resize the images and invert it (black background)
    gray = cv2.resize(255 - gray, (28, 28))

    # save the processed images
    cv2.imwrite("pro-img/image.png", gray)
    """
        all images in the training set have an range from 0-1
        and not from 0-255 so we divide our flatten images
        (a one dimensional vector with our 784 pixels)
        to use the same 0-1 based range
        """
    flatten = gray.flatten() / 255.0
    """
        we need to store the flatten image and generate
        the correct_vals array
        correct_val for the first digit (9) would be
        [0,0,0,0,0,0,0,0,0,1]
        """
    images = flatten

    return images

if __name__ == "__main__":
    network = Net()

    model=Net()
    PATH="/home/eden/newMODEL/results/model.pth"
    model.load_state_dict(torch.load(PATH))
    model.eval()

    Example()
