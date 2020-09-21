import matplotlib as mpl
import numpy as np
from scipy.ndimage import rotate, gaussian_filter
from torch.autograd import Variable
#from torchvision.transforms.functional import rotate

import config as c
from utils import *

mpl.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

GRADIENT_MAP_DIR = './gradient_maps/'


def save_imgs(inputs, grad, cnt):
    print(f"calling save_image(input={inputs.shape}, grad={grad.shape})")
    export_dir = os.path.join(GRADIENT_MAP_DIR, c.modelname)
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)

    for g in range(grad.shape[0]):
        normed_grad = (grad[g] - np.min(grad[g])) / (np.max(grad[g]) - np.min(grad[g]))
        # normed_grad = grad[g]
        print("cnt={:d}/g={:d}: minGrad={:.2e}, maxGrad={:.2e}, max/min={:.2e}".format(
            cnt, g, np.min(grad[g]), np.max(grad[g]), np.max(grad[g])/np.min(grad[g])))
        orig_image = inputs[g]
        for image, file_suffix in [
            (normed_grad, "_gradient_map.png"),
            (orig_image, "_orig.png"),
        ]:
            plt.clf()
            plt.imshow(image)
            #plt.imshow(image, vmin=0, vmax=1e13)
            plt.axis("off")
            plt.savefig(
                os.path.join(export_dir, f"{cnt}_{g}" + file_suffix),
                bbox_inches="tight",
                pad_inches=0,
            )
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
        print(f"i={i}: inputs={inputs.shape}, labels={labels}")
        inputs = Variable(inputs, requires_grad=True)

        emb = model(inputs)
        loss = get_loss(emb, model.nf.jacobian(run_forward=False))
        loss.backward()

        grad = inputs.grad.view(-1, c.n_transforms_test, *inputs.shape[-3:])
        grad = grad[labels > 0]
        if grad.shape[0] == 0:
            continue
        grad = t2np(grad)

        print(f"origina: shape of inputs={inputs.shape}")
        allInputs = inputs.view(c.n_transforms_test, *inputs.shape[-3:])
        inputs = inputs.view(-1, c.n_transforms_test, *inputs.shape[-3:])[:, 0]
       
        print(f"view: shape of inputs={inputs.shape}, allInputs={allInputs.shape}")
        inputs = np.transpose(t2np(inputs[labels > 0]), [0, 2, 3, 1])
        allInputs = np.transpose(t2np(allInputs), [0, 2, 3, 1])
 
        print(f"transpose: shape of inputs={inputs.shape}, allInputs={allInputs.shape}")
        inputs_unnormed = np.clip(inputs * c.norm_std + c.norm_mean, 0, 1)
        print(f"shape of inputs={inputs.shape},inputs_unnormed={inputs_unnormed.shape}")

        images = np.zeros([c.n_transforms_test,480, 270, 3])
        for i_item in range(c.n_transforms_test):
            old_shape = grad[:, i_item].shape
            img = np.reshape(grad[:, i_item], [-1, *grad.shape[-2:]])
            img = np.transpose(img, [1, 2, 0])
            img = np.transpose(rotate(img, degrees[i_item], reshape=False), [2, 0, 1])
            img = gaussian_filter(img, (0, 3, 3))
            grad[:, i_item] = np.reshape(img, old_shape)
            # print(f"shape of img={img.shape}, grad={grad.shape}")
            # PyTorch tensors assume the color channel is the first dimension
            # but matplotlib assumes is the third dimension
            images[i_item, :] = img.transpose((1, 2, 0)) 
        
        #save_imgs(allInputs,images,0)
            

        grad = np.reshape(grad, [grad.shape[0], -1, *grad.shape[-2:]])
        grad_img = np.mean(np.abs(grad), axis=1)
        grad_img_sq = grad_img ** 2
        print(f"shape of grad={grad.shape}, grad_img={grad_img.shape}")
        # print(f"inputs_unnormed={inputs_unnormed}")
        # print(f"grad_img_sq={grad_img_sq}")

        cnt = save_imgs(inputs_unnormed, grad_img_sq, cnt)

        if i == n_batches:
            break

    plt.close()
    testloader.dataset.get_fixed = False
