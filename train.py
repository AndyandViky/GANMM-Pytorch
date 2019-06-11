# encoding: utf-8
"""
@author: andy
@contact: andy_viky@163.com
@github: https://github.com/AndyandViky
@csdn: https://blog.csdn.net/AndyViky
@file: train.py
@time: 2019/6/9 下午8:18
@desc: train
"""

try:
    import os
    import argparse
    from ganmm.datasets import dataset_list
    from ganmm.definitions import RUNS_DIR, DATASETS_DIR
except ImportError as e:
    print(e)
    raise ImportError


def main():
    global args
    parser = argparse.ArgumentParser(description="Convolutional NN Training Script")
    parser.add_argument("-r", "--run_name", dest="run_name", default="ganmm", help="Name of training run")
    parser.add_argument("-n", "--n_epochs", dest="n_epochs", default=200, type=int, help="Number of epochs")
    parser.add_argument("-b", "--batch_size", dest="batch_size", default=64, type=int, help="Batch size")
    parser.add_argument("-p", "--pretrain_iter", dest="pretrain_iter", default=10, type=int, help="pretrain iter")
    parser.add_argument("-s", "--dataset_name", dest="dataset_name", default='mnist', choices=dataset_list,
                        help="Dataset name")
    args = parser.parse_args()

    run_name = args.run_name
    dataset_name = args.dataset_name

    # make directory
    run_dir = os.path.join(RUNS_DIR, dataset_name, run_name)
    data_dir = os.path.join(DATASETS_DIR, dataset_name)
    imgs_dir = os.path.join(run_dir, 'images')
    models_dir = os.path.join(run_dir, 'models')

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(imgs_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # -----train-----
    """
    训练流程：
    初始化：将全部数据均匀分布于 N 个聚类
    预训练：利用每个GAN model自身的数据训练 pretrain_iter 次
    开始EM算法流程：训练分类器和 GAN-model
    """
    # train detail var
    lr = 5e-5 # learning rate
    n_cluster = 10
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    pretrain_iter = args.pretrain_iter # 预训练次数





if __name__ == '__main__':
    main()
