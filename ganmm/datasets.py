# encoding: utf-8
"""
@author: andy
@contact: andy_viky@163.com
@github: https://github.com/AndyandViky
@csdn: https://blog.csdn.net/AndyViky
@file: datasets.py
@time: 2019/6/9 下午8:20
@desc: datasets
"""

try:
    import torchvision.datasets as dset
    import torchvision.transforms as transforms
    import torch

except ImportError as e:
    print(e)
    raise ImportError


DATASET_FN_DICT = {'mnist': dset.MNIST}


dataset_list = DATASET_FN_DICT.keys()


def get_dataset(dataset_name='mnist'):

    if dataset_name in DATASET_FN_DICT:
        return DATASET_FN_DICT[dataset_name]
    else:
        raise ValueError('Invalid dataset, {}, entered. Must be '
                         'in {}'.format(dataset_name, dataset_list))


def get_dataloader(dataset_path='../datasets/mnist', dataset_name='mnist', batch_size=50, train=True):

    dataset = get_dataset(dataset_name)

    dataloader = torch.utils.data.DataLoader(
        dataset(dataset_path, download=True, train=train, transform=transforms.Compose([
            transforms.ToTensor()
        ])),
        batch_size=batch_size, shuffle=True
    )
    return dataloader
