# encoding: utf-8
"""
@author: andy
@contact: andy_viky@163.com
@github: https://github.com/AndyandViky
@csdn: https://blog.csdn.net/AndyViky
@file: model.py
@time: 2019/6/9 下午8:20
@desc: model
"""

try:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import matplotlib.pyplot as plt

    from ganmm.utils import init_weights

except ImportError as e:
    print(e)
    raise ImportError


class Reshape(nn.Module):
    """
    Class for performing a reshape as a layer in a sequential model.
    """

    def __init__(self, shape=[]):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.size(0), *self.shape)

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'shape={}'.format(
            self.shape
        )


class Generator(nn.Module):

    def __init__(self, dim=64, x_shape=(1, 32, 32), verbose=False):
        super(Generator, self).__init__()

        self.dim = dim
        self.ishap = (4 * self.dim, 4, 4)
        self.x_shape = x_shape
        self.verbose = verbose

        self.model = nn.Sequential(
            nn.Linear(128, self.dim * 4 * 4 * 4),
            nn.BatchNorm1d(self.dim * 4 * 4 * 4),
            nn.ReLU(True),
            #
            Reshape(self.ishap),
            #
            nn.ConvTranspose2d(4 * self.dim, 2 * self.dim, 4, stride=2, padding=1),
            nn.BatchNorm2d(2 * self.dim),
            nn.ReLU(True),
            #
            nn.ConvTranspose2d(2 * self.dim, self.dim, 4, stride=2, padding=1),
            nn.BatchNorm2d(self.dim),
            nn.ReLU(True),
            #
            nn.ConvTranspose2d(self.dim, 1, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

        init_weights(self)

        if self.verbose:
            print(self.model)

    def forward(self, x):
        gen_img = self.model(x)
        print(gen_img.size())
        return gen_img.view(x.size(0), *self.x_shape)


from torchvision.utils import save_image
input = 0.75*torch.randn(128, 128)
ge = Generator()
gen_img = ge(input)
print(gen_img)
save_image(gen_img.data[:25],
                   '../gen_%06i.png' %(1),
                   nrow=5, normalize=True)


class Discriminator(nn.Module):

    def __int__(self, dim=64, feature_dim=784, verbose=False):
        super(Discriminator, self).__init__()

        self.dim = dim
        self.feature_dim = feature_dim
        self.verbose = verbose

        self.model = nn.Sequential(

        )

        if self.verbose:
            print(self.model)

    def forward(self, x):
        valid = self.model(x)
        return valid


class Classifier(nn.Module):
    def __int__(self, dim=64, feature_dim=784, verbose=False):
        super(Classifier, self).__init__()

        self.dim = dim
        self.feature_dim = feature_dim
        self.verbose = verbose

        self.model = nn.Sequential(

        )

        if self.verbose:
            print(self.model)

    def forward(self, x):
        pass