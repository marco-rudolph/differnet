'''This is the repo which contains the original code to the WACV 2021 paper
"Same Same But DifferNet: Semi-Supervised Defect Detection with Normalizing Flows"
by Marco Rudolph, Bastian Wandt and Bodo Rosenhahn.
For further information contact Marco Rudolph (rudolph@tnt.uni-hannover.de)'''

import config as c
from train import train
from utils import load_datasets, make_dataloaders

train_set, test_set = load_datasets(c.dataset_path, c.class_name)
train_loader, test_loader = make_dataloaders(train_set, test_set)
model = train(train_loader, test_loader)
