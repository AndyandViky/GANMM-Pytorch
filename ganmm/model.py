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

    def __init__(self, latent_dim=50, x_shape=(1, 28, 28), verbose=False):
        super(Generator, self).__init__()

        self.latent_dim = latent_dim
        self.ishape = (128, 7, 7)
        self.iels = int(np.prod(self.ishape))
        self.x_shape = x_shape
        self.verbose = verbose

        self.model = nn.Sequential(
            nn.Linear(self.latent_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),

            nn.Linear(1024, self.iels),
            nn.BatchNorm1d(self.iels),
            nn.ReLU(True),
            #
            Reshape(self.ishape),
            #
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            #
            nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

        init_weights(self)

        if self.verbose:
            print(self.model)

    def forward(self, x):
        gen_img = self.model(x)
        # print(gen_img.size())
        return gen_img.view(x.size(0), *self.x_shape)


# from torchvision.utils import save_image
from tensorboardX import SummaryWriter

n_sample = 50
latent_dim = 50
input = 0.75*torch.randn(n_sample, latent_dim)
ge = Generator()
gen_imgs = ge(input)

with SummaryWriter(comment='Generator') as w:
    w.add_graph(ge, input)

# save_image(gen_imgs.data[:25],
#                    '../gen_%06i.png' %(1),
#                    nrow=5, normalize=True)


class Discriminator(nn.Module):

    def __init__(self, verbose=False, n_cluster=10):
        super(Discriminator, self).__init__()

        self.channels = 1
        self.n_cluster = n_cluster
        self.cshape = (128, 7, 7)
        self.iels = int(np.prod(self.cshape))
        self.lshape = (self.iels,)
        self.verbose = verbose

        self.model = nn.Sequential(
            nn.Conv2d(self.channels, 64, 4, stride=2, padding=1),
            nn.ReLU(True),

            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(True),

            Reshape(self.lshape),

            nn.Linear(self.iels, 1024),
            nn.ReLU(True),

            nn.Linear(1024, 1),
        )

        if self.verbose:
            print(self.model)

    def forward(self, img):
        valid = self.model(img)
        print(valid)
        return valid

#
# di = Discriminator()
# output = di(gen_imgs)


class Classifier(nn.Module):
    def __init__(self, n_cluster=10, verbose=False):
        super(Classifier, self).__init__()

        self.channels = 1
        self.cshape = (128, 7, 7)
        self.iels = int(np.prod(self.cshape))
        self.lshape = (self.iels,)
        self.n_cluster = n_cluster
        self.verbose = verbose

        self.model = nn.Sequential(
            nn.Conv2d(self.channels, 64, 4, stride=2, padding=1),
            nn.ReLU(True),

            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(True),

            Reshape(self.lshape),

            nn.Linear(self.iels, self.n_cluster),
            nn.Softmax()
        )

        if self.verbose:
            print(self.model)

    def forward(self, x):
        # 输入 x 为 [n_sample, 1, 28, 28]
        result = self.model(x)
        return result


# cl = Classifier()
# cls = cl(gen_imgs)
# print(cls)