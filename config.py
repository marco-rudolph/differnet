'''This file configures the training procedure because handling arguments in every single function is so exhaustive for
research purposes. Don't try this code if you are a software engineer.'''

# device settings
device = 'cpu' # 'cuda' or 'cpu'
import torch
torch.cuda.set_device(0)

# data settings
dataset_path = "dataset"
class_name = "zerobox-2009-5"
modelname = "zerobox-2009-5"

img_size = (448, 448)
img_dims = [3] + list(img_size)
add_img_noise = 0.01

# transformation settings
transf_rotations = True
transf_brightness = 0.0
transf_contrast = 0.0
transf_saturation = 0.0
norm_mean, norm_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

# network hyperparameters
n_scales = 3 # number of scales at which features are extracted, img_size is the highest - others are //2, //4,...
clamp_alpha = 3 # see paper equation 2 for explanation
n_coupling_blocks = 8
# fc_internal = 2048 # number of neurons in hidden layers of s-t-networks
fc_internal = 1536  # number of neurons in hidden layers of s-t-networks
dropout = 0.0 # dropout in s-t-networks
lr_init = 2e-4
n_feat = 256 * n_scales # do not change except you change the feature extractor

# dataloader parameters
n_transforms = 4 # number of transformations per sample in training
n_transforms_test = 16 # number of transformations per sample in testing
batch_size = 4 # actual batch size is this value multiplied by n_transforms(_test)
batch_size_test = batch_size * n_transforms // n_transforms_test

# total epochs = meta_epochs * sub_epochs
# evaluation after <sub_epochs> epochs
meta_epochs = 1
sub_epochs = 8

# output settings
verbose = True
grad_map_viz = True
hide_tqdm_bar = True
save_model = True

target_tpr = 0.85
