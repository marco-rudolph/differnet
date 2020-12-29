import numpy as np
import os
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import alexnet
from torch.autograd import Variable

import config as c
from freia_funcs import permute_layer, glow_coupling_layer, F_fully_connected, ReversibleGraphNet, OutputNode, \
    InputNode, Node

from datetime import datetime
import matplotlib.pyplot as plt
import json
import cv2

WEIGHT_DIR = './weights'
MODEL_DIR = './models'

# copy from https://github.com/pytorch/examples/blob/master/vae/main.py
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# todo: VAE + CNN to generate mask, we need to train the CNN's parameter

class MaskDifferNet(nn.Module):
    def __init__(self):
        super(MaskDifferNet, self).__init__()
        self.differnet = DifferNet()
        self.nf = self.differnet.nf
        self.vae = VAE()

    def forward(self, x):
        y = self.vae(x)

        mask = torch.relu(torch.sign(torch.sigmoid(y[0]) - 0.5))
        y_img = torch.squeeze(y[0].view(x.shape)).permute(2, 1, 0).cpu().detach().numpy()
        cv2.imshow('VAE output', y_img)
        cv2.waitKey(1)

        mask = mask.view(x.shape)
        x_img = torch.squeeze(x).permute(2, 1, 0).cpu().detach().numpy()
        cv2.imshow('original input', x_img)
        cv2.waitKey(1)

        z = x * mask.int().float()
        z_img = torch.squeeze(z).permute(2, 1, 0).cpu().detach().numpy()
        cv2.imshow('original + mask', z_img)
        cv2.waitKey(1)

        output = self.differnet(y[0].view(x.shape))

        return output

def nf_head(input_dim=c.n_feat):
    nodes = list()
    nodes.append(InputNode(input_dim, name='input'))
    for k in range(c.n_coupling_blocks):
        nodes.append(Node([nodes[-1].out0], permute_layer, {'seed': k}, name=F'permute_{k}'))
        # nodes.append(Node([nodes[-1].out0], LUInvertibleMM, {'seed': k}, name=F'permute_{k}'))
        nodes.append(Node([nodes[-1].out0], glow_coupling_layer,
                          {'clamp': c.clamp_alpha, 'F_class': F_fully_connected,
                           'F_args': {'internal_size': c.fc_internal, 'dropout': c.dropout}},
                          name=F'fc_{k}'))
    nodes.append(OutputNode([nodes[-1].out0], name='output'))
    coder = ReversibleGraphNet(nodes)
    return coder


class DifferNet(nn.Module):
    def __init__(self):
        super(DifferNet, self).__init__()
        self.feature_extractor = alexnet(pretrained=True)
        self.nf = nf_head()

    def forward(self, x):
        y_cat = list()

        for s in range(c.n_scales):
            x_scaled = F.interpolate(x, size=c.img_size[0] // (2 ** s)) if s > 0 else x
            feat_s = self.feature_extractor.features(x_scaled)
            y_cat.append(torch.mean(feat_s, dim=(2, 3)))

        y = torch.cat(y_cat, dim=1)
        z = self.nf(y)
        return z


def save_model(model, filename):
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    torch.save(model, os.path.join(MODEL_DIR, filename))


def load_model(filename):
    path = os.path.join(MODEL_DIR, filename)
    model = torch.load(path)
    return model


def save_weights(model, filename):
    if not os.path.exists(WEIGHT_DIR):
        os.makedirs(WEIGHT_DIR)
    torch.save(model.state_dict(), os.path.join(WEIGHT_DIR, filename))


def load_weights(model, filename):
    path = os.path.join(WEIGHT_DIR, filename)
    model.load_state_dict(torch.load(path))
    return model


def save_parameters(model_parameters, filename):
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    with open(MODEL_DIR + '/' + filename + '.json', 'w') as jsonfile:
        jsonfile.write(json.dumps(model_parameters, indent=4))

def save_roc_plot(fpr, tpr, filename):
    plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr.tolist(), tpr.tolist(), color='darkorange',
             lw=lw, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    now = datetime.now()
    dt_string = now.strftime("%Y%m%d%H%M%S")
    # plt.savefig(MODEL_DIR + '/' +filename + '_ROC_' + dt_string + '.jpg')
    plt.savefig(MODEL_DIR + '/' + filename + '_ROC.jpg')