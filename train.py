# encoding: utf-8
"""
@author: andy
@contact: andy_viky@163.com
@github: https://github.com/AndyandViky
@csdn: https://blog.csdn.net/AndyViky
@file: train.py
@time: 2019/6/9 下午8:18
@desc: train
from google.colab import drive
drive.mount('/content/drive/')
4/bgHximcU5YvPiUE75nKKSp4oZH-LLMO_UTcrP-fzn3GDmAnp2qqZHgQ
"""

try:
    import os
    import argparse
    import copy

    import torch
    import torch.nn as nn
    from torchvision.utils import save_image
    import numpy as np
    from ganmm.datasets import dataset_list, get_full_data_loader, get_dataloaders
    from ganmm.definitions import RUNS_DIR, DATASETS_DIR
    from ganmm.model import Generator, Discriminator, Classifier
    from ganmm.utils import calc_gradient_penalty, init_weights, get_fake_imgs, \
        softmax_cross_entropy_with_logits, sample_realimages, save_images, get_performance
except ImportError as e:
    print(e)
    raise ImportError


def main():
    global args
    parser = argparse.ArgumentParser(description="Convolutional NN Training Script")
    parser.add_argument("-r", "--run_name", dest="run_name", default="ganmm", help="Name of training run")
    parser.add_argument("-n", "--n_epochs", dest="n_epochs", default=20000, type=int, help="Number of epochs")
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
    log_path = os.path.join(run_dir, 'logs')

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(imgs_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

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
    load_pre_params = True

    # test detail var
    test_batch_size = 2500

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

    # dataloader
    dataloaders = get_dataloaders(dataset_path=data_dir, dataset_name=dataset_name,
                                  train=pre_train, n_cluster=n_cluster, batch_size=batch_size)
    testdatas = get_full_data_loader(dataset_path=data_dir, dataset_name=dataset_name,
                                  train=False, batch_size=test_batch_size)
    fulldataloader = get_full_data_loader(dataset_path=data_dir, dataset_name=dataset_name,
                                          train=pre_train, batch_size=batch_size)

    # loss
    gen_cost = [0.0 for i in range(10)]
    disc_cost = [0.0 for i in range(10)]

    # param_dict
    gen_params = []
    disc_params = []

    # optimizations
    gen_train_op = []
    disc_train_op = []
    for i in range(n_cluster):
        gen_train_op.append(torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2), weight_decay=decay))
        disc_train_op.append(torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2)))

    # classifier_op = torch.optim.SGD(classifier.parameters(), lr=0.001, momentum=0.9)
    classifier_op = torch.optim.Adam(classifier.parameters(), lr=lr, betas=(b1, b2), weight_decay=decay)

    # ==== pretrain ====
    if load_pre_params:
        # 加载预先训练好的参数
        for i in range(n_cluster):
            if os.path.exists("{}/G{}_params.pkl".format(models_dir, i)) \
                    and os.path.exists("{}/D{}_params.pkl".format(models_dir, i)):
                gen_params.append(torch.load("{}/G{}_params.pkl".format(models_dir, i)))
                disc_params.append(torch.load("{}/D{}_params.pkl".format(models_dir, i)))
            else:
                print('pre-train-params is not exists')
                return
    else:
        print('pretrain......')

        for model_index in range(n_cluster):
            print('start {}'.format(model_index))
            init_weights(generator)
            init_weights(discriminator)
            for it in range(5):
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

                    if i % n_skip_iter == 0:
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
                       '%s/pre_train_gan_%04i.png' % (imgs_dir, model_index),
                       nrow=5, normalize=True)

            torch.save(generator.state_dict(), "{}/G{}_params.pkl".format(models_dir, model_index))
            torch.save(discriminator.state_dict(), "{}/D{}_params.pkl".format(models_dir, model_index))
            gen_params.append(copy.deepcopy(generator.state_dict()))
            disc_params.append(copy.deepcopy(discriminator.state_dict()))

    printP = True
    print('EM-train......')
    for epoch in range(n_epochs):

        if epoch < 500:
            num_choose = batch_size-30
        elif epoch < 1000:
            num_choose = batch_size-15
        elif epoch < 2000:
            num_choose = batch_size-8
        else:
            num_choose = batch_size-2

        # train classifier
        for cls_iter in range(0, 1):
            fake_imgs = get_fake_imgs(netG=generator, n_cluster=n_cluster, \
                                      n_sample=n_sample, latent_dim=latent_dim, G_params=gen_params)

            cls_cost = [0.0] * 10
            for model_index in range(n_cluster):
                classifier.zero_grad()
                cls_target = torch.zeros([batch_size, n_cluster])
                cls_target[:, model_index] = 1
                # cls_target = torch.zeros(batch_size)
                # cls_target = cls_target + model_index
                logits = classifier(fake_imgs[model_index])
                cls_cost[model_index] = torch.mean(softmax_cross_entropy_with_logits(labels=cls_target,
                                                                        logits=logits))
                # cls_cost.to(device)
                cls_cost[model_index].backward(retain_graph=True)
                classifier_op.step()

        # train GAN
        for model_index in range(n_cluster):
            generator.load_state_dict(gen_params[model_index])
            discriminator.load_state_dict(disc_params[model_index])

            generator.train()
            generator.zero_grad()
            discriminator.zero_grad()
            gen_train_op[model_index].zero_grad()

            input = 0.75 * torch.randn(n_sample, latent_dim)
            gen_imgs = generator(input.to(device))

            D_gen = discriminator(gen_imgs)

            real_imgs = sample_realimages(datasets=fulldataloader, classifier=classifier, model_index=model_index,
                                          num_choose=num_choose, batch_size=batch_size)

            real_imgs = real_imgs.to(device)
            D_real = discriminator(real_imgs)

            # train generator
            gen_cost[model_index] = torch.mean(D_gen)
            gen_cost[model_index].backward(retain_graph=True)
            gen_train_op[model_index].step()
            # 深度copy,防止参数被覆盖
            gen_params[model_index] = copy.deepcopy(generator.state_dict())

            # train discriminator
            disc_train_op[model_index].zero_grad()
            disc_cost[model_index] = torch.mean(D_real) - torch.mean(D_gen) + \
                                     calc_gradient_penalty(discriminator, real_imgs, gen_imgs)
            disc_cost[model_index].backward()
            disc_train_op[model_index].step()
            # 深度copy,防止参数被覆盖
            disc_params[model_index] = copy.deepcopy(discriminator.state_dict())

        # test
        if epoch % 50 == 49:
            trn_img, trn_target = next(iter(testdatas))
            trn_img = trn_img.to(device)

            pred_lbl = []
            iter_num = int(np.floor(trn_img.size(0) / 50))
            for i in range(0, iter_num):
                batch = trn_img[50 * i:50 * (i + 1), :]
                _proba = classifier(batch)
                _proba = _proba.detach().cpu().numpy()
                tmp = np.argmax(_proba, axis=1)
                if len(pred_lbl) == 0:
                    pred_lbl = tmp
                else:
                    pred_lbl = np.hstack((pred_lbl, tmp))
            purity, nmi, ari = get_performance(trn_target.numpy(), pred_lbl, n_cluster)
            print("iter={}, purity={:.4f}, nmi={:.4f}, ari={:.4f}".format(
                epoch, purity, nmi, ari
            ))
            logger = open(os.path.join(log_path, "log.txt"), 'a')
            logger.write(
                "iter={}, purity={:.4f}, nmi={:.4f}, ari={:.4f}\n".format(
                    epoch, purity, nmi, ari
                )
            )
            logger.close()

        if epoch % 500 == 499:
            for i in range(n_cluster):
                # cheek GAN picture
                generator.load_state_dict(gen_params[i])
                input = 0.75 * torch.randn(n_sample, latent_dim)
                gen_imgs = generator(input.to(device))
                save_image(gen_imgs.data[:25],
                           '%s/%d_train_gan_%04i.png' % (imgs_dir, epoch, i),
                           nrow=5, normalize=True)

                # cheek classifier picture
                real_imgs = sample_realimages(datasets=fulldataloader, classifier=classifier, model_index=i,
                                              num_choose=batch_size, batch_size=batch_size)
                save_image(real_imgs.data[:25],
                           '%s/%d_train_classifier_%04i.png' % (imgs_dir, epoch, i),
                           nrow=5, normalize=True)

        if epoch % 3000 == 2999:
            cheek_path = os.path.join(models_dir, 'cheekpoint%d' % epoch)
            os.makedirs(cheek_path, exist_ok=True)
            torch.save(classifier.state_dict(), "{}/C_params.pkl".format(cheek_path))
            for i in range(n_cluster):
                # save model
                torch.save(gen_params[i], "{}/G{}_params.pkl".format(cheek_path, i))
                torch.save(disc_params[i], "{}/D{}_params.pkl".format(cheek_path, i))


if __name__ == '__main__':
    main()