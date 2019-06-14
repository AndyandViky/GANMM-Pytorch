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

    import torch
    from torchvision.utils import save_image
    from ganmm.datasets import dataset_list
    from ganmm.definitions import RUNS_DIR, DATASETS_DIR
    from ganmm.model import Generator, Discriminator, Classifier
    from ganmm.datasets import get_dataloaders
    from ganmm.utils import calc_gradient_penalty, init_weights
except ImportError as e:
    print(e)
    raise ImportError


def main():
    global args
    parser = argparse.ArgumentParser(description="Convolutional NN Training Script")
    parser.add_argument("-r", "--run_name", dest="run_name", default="ganmm", help="Name of training run")
    parser.add_argument("-n", "--n_epochs", dest="n_epochs", default=200000, type=int, help="Number of epochs")
    parser.add_argument("-b", "--batch_size", dest="batch_size", default=60, type=int, help="Batch size")
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
    lr = 1e-4 # learning rate
    n_cluster = 10
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    n_skip_iter = 1 # 训练一次generator训练1次discriminator
    n_pre_train = 100
    pre_train = True
    n_sample = batch_size
    latent_dim = 50
    b1 = 0.5
    b2 = 0.9  # 99
    decay = 2.5 * 1e-5

    # test detail var
    test_batch_size = 5000

    cuda = True if torch.cuda.is_available() else False

    generator = Generator(latent_dim=latent_dim)
    discriminator = Discriminator(n_cluster=n_cluster)
    classifier = Classifier(n_cluster=n_cluster)

    # if cuda:
    #     generator.cuda()
    #     discriminator.cuda()
    #     classifier.cuda()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    generator.to(device)
    discriminator.to(device)
    classifier.to(device)

    dataloaders = get_dataloaders(dataset_path=data_dir, dataset_name=dataset_name,
                                  train=pre_train, n_cluster=n_cluster, batch_size=batch_size)
    testdatas = get_dataloaders(dataset_path=data_dir, dataset_name=dataset_name,
                                  train=False, n_cluster=n_cluster, batch_size=test_batch_size)

    # loss
    gen_cost = [0.0 for i in range(10)]
    disc_cost = [0.0 for i in range(10)]

    # param_dict
    gen_param_dict = []
    disc_param_dict = []

    # optimizations
    gen_train_op = []
    disc_train_op = []
    for i in range(n_cluster):
        gen_train_op.append(torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2), weight_decay=decay))
        disc_train_op.append(torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2)))

    # ==== pretrain ====
    print('pretrain......')

    for model_index in range(n_cluster):
        print('start {}'.format(model_index))
        init_weights(generator)
        init_weights(discriminator)
        for iter in range(3):
            for i, (real_imgs, target) in enumerate(dataloaders[model_index]):
                if i == n_pre_train:
                    break
                real_imgs, target = real_imgs.to(device), target.to(device)
                generator.train()
                generator.zero_grad()
                discriminator.zero_grad()
                gen_train_op[model_index].zero_grad()

                input = 0.75 * torch.randn(n_sample, latent_dim)
                gen_imgs = generator(input.to(device))

                D_gen = discriminator(gen_imgs)
                D_real = discriminator(real_imgs)

                if (i % n_skip_iter == 0):
                    gen_cost[model_index] = torch.mean(D_gen)
                    gen_cost[model_index].backward(retain_graph=True)
                    gen_train_op[model_index].step()

                disc_train_op[model_index].zero_grad()

                # Gradient penalty term
                grad_penalty = calc_gradient_penalty(discriminator, real_imgs, gen_imgs)
                disc_cost[model_index] = torch.mean(D_real) - torch.mean(D_gen) + grad_penalty

                disc_cost[model_index].backward()
                disc_train_op[model_index].step()

        save_image(gen_imgs.data[:25],
                   './gen_%04i.png' % (model_index),
                   nrow=5, normalize=True)
        gen_param_dict.append(generator.state_dict())
        disc_param_dict.append(discriminator.state_dict())


if __name__ == '__main__':
    main()