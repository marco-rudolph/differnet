import torch
import torchvision.transforms as transforms
from torchvision.datasets import DatasetFolder
from torchvision.datasets.folder import make_dataset, pil_loader, default_loader, IMG_EXTENSIONS
from torchvision.transforms.functional import rotate

import config as c


def fixed_rotation(self, sample, degrees):
    cust_rot = lambda x: rotate(x, degrees, False, False, None)
    augmentative_transforms = [cust_rot]
    if c.transf_brightness > 0.0 or c.transf_contrast > 0.0 or c.transf_saturation > 0.0:
        augmentative_transforms += [
            transforms.ColorJitter(brightness=c.transf_brightness, contrast=c.transf_contrast,
                                   saturation=c.transf_saturation)]
    tfs = [transforms.Resize(c.img_size)] + augmentative_transforms + [transforms.ToTensor(),
                                                                       transforms.Normalize(c.norm_mean,
                                                                                            c.norm_std)]
    return transforms.Compose(tfs)(sample)


class DatasetFolderMultiTransform(DatasetFolder):
    """Adapts class DatasetFolder of PyTorch in a way that one sample is transformed several times.
    Args:
        n_transforms (int): number of transformations per sample
        all others: see torchvision.datasets.DatasetFolder
    """

    def __init__(self, root, loader, extensions=None, transform=None,
                 target_transform=None, is_valid_file=None, n_transforms=1):
        super(DatasetFolderMultiTransform, self).__init__(root, loader, extensions=extensions, transform=transform,
                                                          target_transform=target_transform)
        try:
            classes, class_to_idx = self.find_classes(self.root)
        except:
            classes, class_to_idx = self._find_classes(self.root)
        if is_valid_file is not None:
            extensions = None
        self.samples = make_dataset(self.root, class_to_idx, extensions)
        self.n_transforms = n_transforms
        self.get_fixed = False  # set to true if the rotations should be fixed and regularly over 360 degrees
        self.fixed_degrees = [i * 360.0 / n_transforms for i in range(n_transforms)]

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            samples = list()
            for i in range(self.n_transforms):
                if self.get_fixed:
                    samples.append(fixed_rotation(self, sample, self.fixed_degrees[i]))
                else:
                    samples.append(self.transform(sample))
            samples = torch.stack(samples, dim=0)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return samples, target


class ImageFolderMultiTransform(DatasetFolderMultiTransform):
    """Adapts class ImageFolder of PyTorch in a way that one sample can be transformed several times.
    Args:
        n_transforms (int): number of transformations per sample
        all others: see ImageFolder
    """

    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader, is_valid_file=None, n_transforms=c.n_transforms):
        super(ImageFolderMultiTransform, self).__init__(root, loader, IMG_EXTENSIONS,
                                                        transform=transform,
                                                        target_transform=target_transform,
                                                        is_valid_file=is_valid_file, n_transforms=n_transforms)
        self.imgs = self.samples
