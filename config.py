'''This file configures the training procedure because handling arguments in every single function is so exhaustive for
research purposes. Don't try this code if you are a software engineer.'''
import torch

# device settings
# device = 'cuda' # or 'cpu'
device = 'cpu' # or 'cuda'

#torch.cuda.set_device(0)

# data settings
dataset_path = "zerobox_dataset"
class_name = "zerobox_class"
modelname = "zerobox_test"

# img_size = (448, 448)
img_size = (480, 270)
img_dims = [3] + list(img_size)
add_img_noise = 0.01

# transformation settings
transf_rotations = False
transf_brightness = 0.0
transf_contrast = 0.0
transf_saturation = 0.0
norm_mean, norm_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

# network hyperparameters
n_scales = 3 # number of scales at which features are extracted, img_size is the highest - others are //2, //4,...
clamp_alpha = 3 # see paper equation 2 for explanation
n_coupling_blocks = 8
fc_internal = 2048 # number of neurons in hidden layers of s-t-networks
dropout = 0.0 # dropout in s-t-networks
lr_init = 2e-4
n_feat = 256 * n_scales # do not change except you change the feature extractor

# dataloader parameters
n_transforms = 1 # number of transformations per sample in training
n_transforms_test = 64 # number of transformations per sample in testing
batch_size = 1 # actual batch size is this value multiplied by n_transforms(_test)
batch_size_test = 1 # batch_size * n_transforms // n_transforms_test

# total epochs = meta_epochs * sub_epochs
# evaluation after <sub_epochs> epochs
meta_epochs = 10
sub_epochs = 8

# output settings
verbose = True
grad_map_viz = True
hide_tqdm_bar = True
save_model = True
