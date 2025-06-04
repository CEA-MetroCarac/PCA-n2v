from tifffile import imread
import numpy as np
from numpy import random
import os
import matplotlib.pyplot as plt
from scipy import sparse

def subset_image(
  image:np.ndarray,
  noise_level: float,
):
    image = image.astype('int32')
    simulated_image = np.random.binomial(image, noise_level)
    return simulated_image

nx = 512
ny = 512
nmz = 100
shape = (nx, ny, nmz)

def create_synthetic (
  dir_gt_img: str,
  x: int,
  y: int,
  n_mz: int = 100,
  noise_init: int = 150000,
):
    """Create a synthetic ground truth and noisy datasets based on the images in dir_gt_img."""
    list_gt_img = os.listdir(dir_gt_img)
    shape = (x, y, n_mz)    
    stack_gt = np.empty(shape)
    stack_simulated = np.empty(shape)
    for mz in range(n_mz):
        gt_mz_img = np.empty((x, y))
    for f_gt_img in list_gt_img:
        if '.tif' in f_gt_img:
            gt_img = imread(os.path.join(dir_gt_img, f_gt_img))
            chosen = np.random.random()
            if chosen>0.5:
                factor = np.power((np.random.randint(0,100)),2)
            else:
                factor = 0
            gt_mz_img = gt_mz_img + factor*gt_img
    noise_level = np.random.random()/noise_init
    simulated = subset_image(gt_mz_img, noise_level)
    stack_gt[...,mz] = gt_mz_img
    stack_simulated[...,mz] = simulated

    stack_simulated = stack_simulated.astype('int32')
    return stack_gt, stack_simulated
