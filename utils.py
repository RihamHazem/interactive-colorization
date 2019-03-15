import numpy as np
import torch
from skimage import io, transform, color
import cv2
from matplotlib import pyplot as plt

from torch.utils.data.sampler import SubsetRandomSampler


def rgb2lab(rgb_imgs):
    # rgb_imgs shape: (N, H, W, C)
    # returned images shape: (N, C, H, W)
    lab_imgs = np.zeros(rgb_imgs.shape)
    i = 0
    for rgb_img in rgb_imgs:
        lab_imgs[i, :, :, :] = color.rgb2lab(rgb_img / 255.)
        i += 1

    # Change the shape of images from (N, H, W, C) to (N, C, H, W)
    return np.rollaxis(lab_imgs, 3, 1)


def lab2rgb(imgs):
    # rgb_imgs shape: (N, C, H, W)
    # returned images shape: (N, C, H, W)

    imgs = np.rollaxis(imgs, 1, 4)
    rgb_imgs = np.zeros(imgs.shape)
    i = 0
    for lab_img in imgs:
        rgb_imgs[i, :, :, :] = color.lab2rgb(lab_img)
        i += 1
    rgb_imgs = rgb_imgs * 255
    # Change the shape of images from (N, H, W, C) to (N, C, H, W)
    return torch.from_numpy(np.rollaxis(rgb_imgs, 3, 1))


def rgb2gray(rgb_imgs):
    N, H, W, C = rgb_imgs.shape
    gray_imgs = np.zeros((N, 1, H, W))
    i = 0
    for rgb_img in rgb_imgs:
        gray_imgs[i] = color.rgb2gray(rgb_img / 255.)
        i += 1
    return gray_imgs


def put_hints(AB, BW):
    N, C, H, W = AB.shape
    # print(BW.shape)
    hints_imgs = np.zeros(AB.shape)
    mask_imgs = np.zeros(BW.shape)

    for i in range(N):
        points = np.random.geometric(0.125)
        # print("Points: ", points)

        for z in range(points):
            patch_size = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9])

            h = int(np.clip(np.random.normal((H - patch_size + 1) / 2., (H - patch_size + 1) / 4.), 0, H - patch_size))
            w = int(np.clip(np.random.normal((H - patch_size + 1) / 2., (H - patch_size + 1) / 4.), 0, H - patch_size))
            # print("Point Size: ", patch_size, " H: ", h, " W: ", w)
            hints_imgs[i, :, h:h + patch_size, w:w + patch_size] = AB[i, :, h:h + patch_size, w:w + patch_size]

            mask_imgs[i, :, h:h + patch_size, w:w + patch_size] = 1.
    return hints_imgs, mask_imgs


def transformImage(images):
    new_np = np.zeros((images.shape[0], 256, 256, images.shape[3]))
    for i in range(len(new_np)):
        new_np[i] = cv2.resize(images[i], (256, 256), interpolation=cv2.INTER_CUBIC)
    return new_np


def convert_to_numpy(ic):
    # converts images from ImageCollection type to numpy array type
    return ic.concatenate()


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def preprocessing(data):
    rgb_images = data.data.numpy()  # shape (N, H, W, 3)
    # print("converted")
    rgb_images = transformImage(rgb_images)  # shape (N, H, W, 3)
    # print(rgb_images.shape)
    # print("transdorm")

    gray_images = rgb2gray(rgb_images)  # shape (N, 1, H, W)
    # print("rgb2gray")
    # print(gray_images.shape)
    lab_images = rgb2lab(rgb_images)  # shape (N, 3, H, W)
    # print(lab_images.shape)
    # print("rgb2lab")
    ab_images = lab_images[:, 1:, :, :]  # shape (N, 2, H, W)
    hints_images, mask_images = put_hints(ab_images, gray_images)
    #   rgb_imgs = lab2rgb(lab_images)
    #   io.imshow(rgb_imgs[0])
    #   plt.show()
    # print("hints_images")
    result = torch.cat((torch.from_numpy(gray_images), torch.from_numpy(hints_images), torch.from_numpy(mask_images)),
                       dim=1)
    # rgb_images = np.rollaxis(rgb_images, 3, 1)
    #   print(rgb_images.shape)
    return result.float(), torch.from_numpy(ab_images)  # shape (N, 4, H, W)
