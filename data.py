import numpy as np
import torch
from skimage import io
import glob
from PIL import Image

from torch.utils.data.sampler import SubsetRandomSampler


def get_samplers(dataset_size):
    indices = list(range(dataset_size))
    validation_split = .2
    shuffle_dataset = True
    random_seed = 42
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    return train_sampler, valid_sampler


def load_data():
    filenames = []
    for filename in glob.glob('try/*.jpg'):  # assuming gif
        filenames.append(filename)
    images = io.imread_collection(filenames)
    return images
