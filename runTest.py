'''This is the repo which contains the original code to the WACV 2021 paper
"Same Same But DifferNet: Semi-Supervised Defect Detection with Normalizing Flows"
by Marco Rudolph, Bastian Wandt and Bodo Rosenhahn.
For further information contact Marco Rudolph (rudolph@tnt.uni-hannover.de)'''

import config as c
from train import train
from utils import load_datasets, make_dataloaders
from model import load_model, load_weights
import numpy as np
import torch
from train import Score_Observer
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import time
from utils import *
from localization import export_gradient_maps

def test(model, test_loader):
    print("Running test")
    score_obs = Score_Observer('AUROC')
    # evaluate
    model.eval()
    if c.verbose:
        print('\nCompute loss and scores on test set:')
    test_loss = list()
    test_z = list()
    test_labels = list()
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader, disable=c.hide_tqdm_bar)):
            inputs, labels = preprocess_batch(data)
            z = model(inputs)
            loss = get_loss(z, model.nf.jacobian(run_forward=False))
            test_z.append(z)
            test_loss.append(t2np(loss))
            test_labels.append(t2np(labels))

    test_loss = np.mean(np.array(test_loss))
    if c.verbose:
        print('{:d} \t test_loss: {:.4f}'.format(test_loss))

    test_labels = np.concatenate(test_labels)
    is_anomaly = np.array([0 if l == 0 else 1 for l in test_labels])

    z_grouped = torch.cat(test_z, dim=0).view(-1, c.n_transforms_test, c.n_feat)
    anomaly_score = t2np(torch.mean(z_grouped ** 2, dim=(-2, -1)))
    score_obs.update(roc_auc_score(is_anomaly, anomaly_score), epoch,
                    print_score=c.verbose or epoch == c.meta_epochs - 1)

    # if c.grad_map_viz:
    #     export_gradient_maps(model, test_loader, optimizer, -1)

train_set, test_set = load_datasets(c.dataset_path, c.class_name)
_, test_loader = make_dataloaders(train_set, test_set)
time_start = time.time()
model = load_model(c.modelname)
load_weights(model, c.modelname)
test(model, test_loader)
time_end = time.time()
time_c = time_end - time_start  # 运行所花时间
print("time cost: {:f} s".format(time_c))

