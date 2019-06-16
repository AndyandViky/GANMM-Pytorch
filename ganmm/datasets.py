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


def _get_dataset(dataset_name='mnist'):

    if dataset_name in DATASET_FN_DICT:
        return DATASET_FN_DICT[dataset_name]
    else:
        raise ValueError('Invalid dataset, {}, entered. Must be '
                         'in {}'.format(dataset_name, dataset_list))


# 均匀划分全部数据
def _spilit_mnistdata(dataset_path='../datasets/mnist', dataset_name='mnist', train=True, n_cluster=10):
    dataset = _get_dataset(dataset_name)

    # get 60000 data
    result = dataset(dataset_path, download=True, train=train, transform=transforms.Compose([
        transforms.ToTensor()
    ]))

    s_datas = []
    epoch_size = int(result.data.size(0) / n_cluster)
    lengths = []
    for i in range(n_cluster):
        if i != n_cluster-1:
            lengths.append(epoch_size)
        else:
            lengths.append(
                result.data.size(0) - (n_cluster-1) * epoch_size
            )

    s_datas = torch.utils.data.random_split(result, lengths)

    return s_datas


# 获取单个dataloader
def _get_dataloader(dataset, batch_size=50):

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size, shuffle=True
    )
    return dataloader


# 获取全部的dataloader
def get_dataloaders(dataset_path='../datasets/mnist',
                    dataset_name='mnist', train=True, n_cluster=10, batch_size=50):
    s_datas = _spilit_mnistdata(dataset_path=dataset_path, dataset_name=dataset_name, train=train, n_cluster=n_cluster)

    s_dataloader = []
    for item in s_datas:
        s_dataloader.append(_get_dataloader(dataset=item, batch_size=batch_size))

    return s_dataloader


# 获取整个数据集的 loader
def get_full_data_loader(dataset_path='../datasets/mnist',
                    dataset_name='mnist', train=True, batch_size=50):
    dataset = _get_dataset(dataset_name)

    loader = torch.utils.data.DataLoader(
        dataset(dataset_path, download=True, train=train, transform=transforms.Compose([
            transforms.ToTensor()
        ])),
        batch_size=batch_size,
        shuffle=True
    )

    return loader