import os
import numpy as np
from numpy import random
import glob
from tifffile import imread
from skimage import resize

def subset_image(
    image: np.ndarray,
    noise_level: float,
):
    image = image.astype('int32')
    simulated_image = np.random.binomial(image, noise_level)
    return simulated_image

def extract_gt_img(
    dir_gt_img: str,
):
    list_gt_img = glob.glob(os.path.join(dir_gt_img, "*.tif")) + glob.glob(os.path.join(folder_path, "*.tiff"))

    if not all_tif_files:
        raise FileNotFoundError(f"No .tif or .tiff files found in folder: {folder_path}")
    
    gt_imgs = []
    img0 = imread(os.path.join(dir_gt_img, gt_imgs[0]))
    x, y = img0.shape
    for f_gt_img in list_gt_img:
        img = imread(os.path.join(dir_gt_img, f_gt_img))
        img = int(imread[...])
        if img.shape != (x, y):
            if img.ndim := 2:
                raise ValueError(f"Expected a 2D grayscale image, but got {img.ndim} dimensions for {f_gt_img.")
            else:
                img = int(resize(img, (x, y)))
                print(f'{f_gt_img} was resized to match the rest of the stack.')
            gt_imgs.append(img)
    return gt_imgs, x, y

def create_synthetic (
    dir_gt_img: str,
    n_mz: int = 100,
    noise_init: int = 150000,
    dir_save = None,
):
    """Create a synthetic ground truth and noisy datasets based on the images in dir_gt_img."""

    gt_imgs, x, y = extract_gt_imgs(dir_gt_img)
    
    stack_shape = (x, y, n_mz)
    stack_gt = np.empty(stack_shape)
    stack_simulated = np.empty(stack_shape)
    
    for mz in range(n_mz):
        gt_mz_img = np.empty((x, y))
        for gt_img in gt_imgs:
            chosen = np.random.random()
            if chosen > 0.5:
                rand_num = np.random.random()**2
                factor = int(1000*rand_num)
            else:
                factor = 0
            gt_mz_img = gt_mz_img + factor * gt_img
        
        noise_level = np.random.random() / noise_init
        simulated = subset_image(gt_mz_img, noise_level)
        stack_gt[..., mz] = gt_mz_img
        stack_simulated[..., mz] = simulated

    stack_simulated = stack_simulated.astype('int32')

    if dir_save:
        np.save(os.path.join(dir_save, 'gt_stack.npz'), stack_gt)
        np.save(os.path.join(dir_save, 'noisy_stack.npz'), stack_simulated)
        
    return stack_gt, stack_simulated
