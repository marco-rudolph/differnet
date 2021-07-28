import numpy as np
from torch.autograd import Variable
import config as c
from utils import *
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from scipy.ndimage import rotate, gaussian_filter

GRADIENT_MAP_DIR = './gradient_maps/'


def save_imgs(inputs, grad, cnt):
    export_dir = os.path.join(GRADIENT_MAP_DIR, c.modelname)
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)

    for g in range(grad.shape[0]):
        normed_grad = (grad[g] - np.min(grad[g])) / (
                np.max(grad[g]) - np.min(grad[g]))
        orig_image = inputs[g]
        for image, file_suffix in [(normed_grad, '_gradient_map.png'), (orig_image, '_orig.png')]:
            plt.clf()
            plt.imshow(image)
            plt.axis('off')
            plt.savefig(os.path.join(export_dir, str(cnt) + file_suffix), bbox_inches='tight', pad_inches=0)
        cnt += 1
    return cnt


def export_gradient_maps(model, testloader, optimizer, n_batches=1):
    plt.figure(figsize=(10, 10))
    testloader.dataset.get_fixed = True
    cnt = 0
    degrees = -1 * np.arange(c.n_transforms_test) * 360.0 / c.n_transforms_test

    # TODO n batches
    for i, data in enumerate(tqdm(testloader, disable=c.hide_tqdm_bar)):
        optimizer.zero_grad()
        inputs, labels = preprocess_batch(data)
        inputs = Variable(inputs, requires_grad=True)

        emb = model(inputs)
        loss = get_loss(emb, model.nf.jacobian(run_forward=False))
        loss.backward()

        grad = inputs.grad.view(-1, c.n_transforms_test, *inputs.shape[-3:])
        grad = grad[labels > 0]
        if grad.shape[0] == 0:
            continue
        grad = t2np(grad)

        inputs = inputs.view(-1, c.n_transforms_test, *inputs.shape[-3:])[:, 0]
        inputs = np.transpose(t2np(inputs[labels > 0]), [0, 2, 3, 1])
        inputs_unnormed = np.clip(inputs * c.norm_std + c.norm_mean, 0, 1)

        for i_item in range(c.n_transforms_test):
            old_shape = grad[:, i_item].shape
            img = np.reshape(grad[:, i_item], [-1, *grad.shape[-2:]])
            img = np.transpose(img, [1, 2, 0])
            img = np.transpose(rotate(img, degrees[i_item], reshape=False), [2, 0, 1])
            img = gaussian_filter(img, (0, 3, 3))
            grad[:, i_item] = np.reshape(img, old_shape)

        grad = np.reshape(grad, [grad.shape[0], -1, *grad.shape[-2:]])
        grad_img = np.mean(np.abs(grad), axis=1)
        grad_img_sq = grad_img ** 2

        cnt = save_imgs(inputs_unnormed, grad_img_sq, cnt)

        if i == n_batches:
            break

    plt.close()
    testloader.dataset.get_fixed = False
