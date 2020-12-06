import config as c
from train import *
from utils import load_datasets, make_dataloaders
import time
import gc
import json

_, _, predict_set = load_datasets(c.dataset_path, 'predict', test=True)
_, _, predict_loader = make_dataloaders(None, None, predict_set, test=True)

model = torch.load("models/" + c.modelname + "", map_location=torch.device('cpu'))

with open('models/' + c.modelname + '.json') as jsonfile:
    model_parameters = json.load(jsonfile)

time_start = time.time()
predict(model, model_parameters, predict_loader)
time_end = time.time()
time_c = time_end - time_start
print("predicting time cost: {:f} s".format(time_c))

