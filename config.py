'''This file configures the training procedure because handling arguments in every single function is so exhaustive for
research purposes. Don't try this code if you are a software engineer.'''

# data extraction settings
num_videos = 21
save_cropped_image_to = "dataset/zerobox-2010-1/"
save_original_image_to = "dataset/zerobox-2010-1-original/"

# device settings
device = 'cuda' # 'cuda' or 'cpu'
import torch
torch.cuda.set_device(0)

# data settings
dataset_path = "dataset"
class_name = "Experiment_3.2"
modelname = "Experiment_3.2_10epoch_239tainingdata_0.5BCS"

img_size = (448, 448)
img_dims = [3] + list(img_size)
add_img_noise = 0.01

# transformation settings
transf_rotations = True
transf_brightness = 0.5
transf_contrast = 0.5
transf_saturation = 0.5
norm_mean, norm_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

rotation_degree = 0
crop_top = 0.10
crop_left = 0.10
crop_bottom = 0.10
crop_right = 0.10

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
n_transforms_test = 1 # number of transformations per sample in testing
batch_size = 4 # actual batch size is this value multiplied by n_transforms(_test)
batch_size_test = 1

# total epochs = meta_epochs * sub_epochs
# evaluation after <sub_epochs> epochs
meta_epochs = 10
sub_epochs = 8

# output settings
verbose = True
grad_map_viz = True
hide_tqdm_bar = True
save_model = True
save_transformed_image = True
visualization = False
frame_name_is_given = False
target_tpr = 0.85
