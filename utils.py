import os
import torch
from torchvision import datasets, transforms

import config as c
from multi_transform_loader import ImageFolderMultiTransform

import cv2
import numpy as np
from datetime import datetime

TRANSFORM_DIR = "./transform/"

def TransformShow(name="img", wait=100):
    def transform_show(img):
        # path = "transform/"
        # now = datetime.now()
        # dt_string = now.strftime("%d%m%Y%H%M%S")
        # cv2.imwrite(path + 'all_transform_' + dt_string + '.jpg', np.array(img))
        # cv2.imshow(name, np.array(img))
        # cv2.waitKey(wait)
        return img

    return transform_show

def cropImage():
    def crop_image(img):
        x,y,w,h = shrinkEdges(img.size)
        rs = transforms.functional.crop(img,y,x,h,w)

        if not os.path.exists(TRANSFORM_DIR):
            os.makedirs(TRANSFORM_DIR)

        now = datetime.now()
        dt_string = now.strftime("%d%m%Y%H%M%S")
        if(c.save_transformed_image):
            cv2.imwrite(TRANSFORM_DIR + 'transform_' + dt_string + '.jpg', np.array(rs))
        return rs

    return crop_image

def shrinkEdges(img_size):
    width, height = img_size
    shrink_scale_top = c.crop_top
    shrink_scale_bot = c.crop_bottom
    shrink_scale_left = c.crop_left
    shrink_scale_right = c.crop_right
    left_reduction = shrink_scale_left * width
    right_reduction = shrink_scale_right * width
    top_reduction = shrink_scale_top * height
    bot_reduction = shrink_scale_bot * height
    new_height = int(height - top_reduction - bot_reduction)
    new_width = int(width - left_reduction - right_reduction)
    new_ul_x = int(left_reduction)
    new_ul_y = int(top_reduction)
    # print(
    #    f"shrinking {0, 0, width, height} to {new_ul_x, new_ul_y, new_width, new_height}"
    # )
    return new_ul_x, new_ul_y, new_width, new_height


def t2np(tensor):
    '''pytorch tensor -> numpy array'''
    return tensor.cpu().data.numpy() if tensor is not None else None


def get_loss(z, jac):
    '''check equation 4 of the paper why this makes sense - oh and just ignore the scaling here'''
    return torch.mean(0.5 * torch.sum(z ** 2, dim=(1,)) - jac) / z.shape[1]


def load_datasets(dataset_path, class_name, test=False):
    '''
    Expected folder/file format to find anomalies of class <class_name> from dataset location <dataset_path>:

    train data:

            dataset_path/class_name/train/good/any_filename.png
            dataset_path/class_name/train/good/another_filename.tif
            dataset_path/class_name/train/good/xyz.png
            [...]

    test data:

        'normal data' = non-anomalies

            dataset_path/class_name/test/good/name_the_file_as_you_like_as_long_as_there_is_an_image_extension.webp
            dataset_path/class_name/test/good/did_you_know_the_image_extension_webp?.png
            dataset_path/class_name/test/good/did_you_know_that_filenames_may_contain_question_marks????.png
            dataset_path/class_name/test/good/dont_know_how_it_is_with_windows.png
            dataset_path/class_name/test/good/just_dont_use_windows_for_this.png
            [...]

        anomalies - assume there are anomaly classes 'crack' and 'curved'

            dataset_path/class_name/test/crack/dat_crack_damn.png
            dataset_path/class_name/test/crack/let_it_crack.png
            dataset_path/class_name/test/crack/writing_docs_is_fun.png
            [...]

            dataset_path/class_name/test/curved/wont_make_a_difference_if_you_put_all_anomalies_in_one_class.png
            dataset_path/class_name/test/curved/but_this_code_is_practicable_for_the_mvtec_dataset.png
            [...]
    '''

    def target_transform(target):
        return class_perm[target]

    data_dir_train = os.path.join(dataset_path, class_name, 'train')
    data_dir_validate = os.path.join(dataset_path, class_name, 'validate')
    data_dir_test = os.path.join(dataset_path, class_name, 'test')

    classes = os.listdir(data_dir_validate)
    if 'good' not in classes:
        print('There should exist a subdirectory "good". Read the doc of this function for further information.')
        exit()
    classes.sort()
    class_perm = list()
    class_idx = 1
    for cl in classes:
        if cl == 'good':
            class_perm.append(0)
        else:
            class_perm.append(class_idx)
            class_idx += 1

    augmentative_transforms = []
    if c.transf_rotations:
        augmentative_transforms += [transforms.RandomRotation(c.rotation_degree)]
    if c.transf_brightness > 0.0 or c.transf_contrast > 0.0 or c.transf_saturation > 0.0:
        augmentative_transforms += [transforms.ColorJitter(brightness=c.transf_brightness, contrast=c.transf_contrast,
                                                           saturation=c.transf_saturation)]

    tfs = [cropImage(), transforms.Resize(c.img_size)] \
          + augmentative_transforms + [ TransformShow("Transformed Image", 10), transforms.ToTensor(), transforms.Normalize(c.norm_mean, c.norm_std)]

    transform_train = transforms.Compose(tfs)

    trainset = None
    validateset = None
    testset = None
    if test == False:
        trainset = ImageFolderMultiTransform(data_dir_train, transform=transform_train, n_transforms=c.n_transforms)
        validateset = ImageFolderMultiTransform(data_dir_validate, transform=transform_train, target_transform=target_transform,
                                        n_transforms=c.n_transforms_test)
    else:
        testset = ImageFolderMultiTransform(data_dir_test, transform=transform_train, target_transform=target_transform,
                                        n_transforms=c.n_transforms_test)

    return trainset, validateset, testset


def make_dataloaders(trainset, validateset, testset, test=False):
    trainloader = None
    validateloader = None
    testloader = None
    if test == False:
        trainloader = torch.utils.data.DataLoader(trainset, pin_memory=True, batch_size=c.batch_size, shuffle=True,
                                                  drop_last=False)
        validateloader = torch.utils.data.DataLoader(validateset, pin_memory=True, batch_size=c.batch_size, shuffle=True,
                                                  drop_last=False)
    else:
        testloader = torch.utils.data.DataLoader(testset, pin_memory=True, batch_size=c.batch_size_test, shuffle=False,
                                                 drop_last=False)

    return trainloader, validateloader, testloader


def preprocess_batch(data):
    '''move data to device and reshape image'''
    inputs, labels = data
    #print(f"begin: size of inputs={inputs.size()}")
    inputs, labels = inputs.to(c.device), labels.to(c.device)
    #print(f"to: size of inputs={inputs.size()}")
    inputs = inputs.view(-1, *inputs.shape[-3:])
    #print(f"view: size of inputs={inputs.size()}")
    return inputs, labels
