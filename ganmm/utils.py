# encoding: utf-8
"""
@author: andy
@contact: andy_viky@163.com
@github: https://github.com/AndyandViky
@csdn: https://blog.csdn.net/AndyViky
@file: utils.py
@time: 2019/6/9 下午8:20
@desc: utils function
"""

try:
    import torch
    import numpy as np
    from torch.autograd import Variable
    from torch.autograd import grad as torch_grad
    import torch.nn.functional as F
    import torch.nn as nn
    from torchvision.utils import save_image

except ImportError as e:
    print(e)
    raise ImportError


def init_weights(net):

    for m in net.modules():
        if isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()


def calc_gradient_penalty(netD, real_data, generated_data):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # GP strength
    LAMBDA = 10

    b_size = real_data.size()[0]

    # Calculate interpolation
    alpha = torch.rand(b_size, 1, 1, 1)
    alpha = alpha.expand_as(real_data)
    alpha = alpha.to(device)

    interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
    interpolated = Variable(interpolated, requires_grad=True)
    interpolated = interpolated.to(device)

    # Calculate probability of interpolated examples
    prob_interpolated = netD(interpolated)

    # Calculate gradients of probabilities with respect to examples
    gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                           grad_outputs=torch.ones(prob_interpolated.size()).to(device),
                           create_graph=True, retain_graph=True)[0]

    # Gradients have shape (batch_size, num_channels, img_width, img_height),
    # so flatten to easily take norm per example in batch
    gradients = gradients.view(b_size, -1)

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

    # Return gradient penalty
    return LAMBDA * ((gradients_norm - 1) ** 2).mean()


def get_fake_imgs(netG, n_cluster, n_sample, latent_dim, G_params):

    gen_imgs = []
    for i in range(n_cluster):
        init_weights(netG)

        input = 0.75 * torch.randn(n_sample, latent_dim)
        # input.to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))

        netG.load_state_dict(G_params[i])
        gen_imgs.append(netG(input.cuda()))
    return gen_imgs


def softmax_cross_entropy_with_logits(labels, logits, dim=-1):

    loss = torch.sum(- labels.cuda() * F.log_softmax(logits, -1), -1)
    return loss


def save_images(gen_imgs, imags_path, images_name):
    save_image(gen_imgs.data[:25],
               '%s/%s.png' % (imags_path, images_name),
               nrow=5, normalize=True)


def sample_realimages(datasets, classifier, model_index, num_choose, batch_size=60, feature_dim=[1, 1, 28, 28]):
    '''
    动态从全体数据中筛选某一领域的真实数据
    :return: images
    '''
    _chosen_data = []
    _rest_data = []
    _rest_data_proba = []
    for i, (_data, _targets) in enumerate(datasets):
        _proba = classifier(_data.cuda())
        _data = _data.numpy()
        _proba = _proba.detach().cpu().numpy()
        # record choosen data
        tmp = []
        idx = np.argmax(_proba, axis=1)
        if (idx == model_index).any():
            tmp = _data[idx == model_index, :]
        else:
            idx = np.argmax(_proba, axis=0)
            tmp = _data[idx[model_index], :]
            tmp = tmp.reshape(feature_dim)
        if len(_chosen_data):
            _chosen_data = np.vstack((_chosen_data, tmp))
        else:
            _chosen_data = tmp
        # record rest data
        tmp = []
        idx = np.argmax(_proba, axis=1)
        if (idx != model_index).any():
            tmp = _data[idx != model_index, :]
            tmp_proba = _proba[idx != model_index, :]
        else:
            idx = np.argmin(_proba, axis=0)
            tmp = _data[idx[model_index], :]
            tmp = tmp.reshape(feature_dim)
            tmp_proba = _proba[idx[model_index], :]
        if len(_rest_data):
            _rest_data = np.vstack((_rest_data, tmp))
        else:
            _rest_data = tmp
        if len(_rest_data_proba):
            _rest_data_proba = np.vstack((_rest_data_proba, tmp_proba))
        else:
            _rest_data_proba = tmp_proba

        if _chosen_data.shape[0] >= num_choose and _rest_data.shape[0] >= batch_size - num_choose:
            break

    _chosen_data = np.vstack((_chosen_data[0:num_choose, :],
                              sample_restdata(_rest_data, _rest_data_proba, model_index,
                                                  batch_size - num_choose)))

    return torch.from_numpy(_chosen_data)


def sample_restdata(rest_data, rest_data_proba, model_index, size):
    P = rest_data_proba[:, model_index]
    P = P / P.sum()
    sample_idx = np.random.choice(len(P), size, replace=False, p=P)

    return rest_data[sample_idx, :]

