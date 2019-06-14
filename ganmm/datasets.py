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
    datas = result.data

    # 均匀划分到 n_cluster 个 cluster
    epoch_size = int(datas.size(0) / n_cluster)
    def getEpoch(num):
        """
        获取单个 epoch 的数据
        :param num:
        :return: epoch data
        """
        return datas[int(num * epoch_size): int((num+1) * epoch_size)]

    s_datas = []
    for i in range(n_cluster):
        s_datas.append(getEpoch(i))

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
    s_dataloader = []
    s_datas = _spilit_mnistdata(dataset_path=dataset_path, dataset_name=dataset_name, train=train, n_cluster=n_cluster)

    for item in s_datas:
        s_dataloader.append(_get_dataloader(dataset=item, batch_size=batch_size))

    return s_dataloader