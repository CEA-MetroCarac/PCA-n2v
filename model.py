import os
import pathlib
from typing import Tuple, Optional
import numpy as np
import tensorflow as tf
import pandas as pd

from n2v.internals.N2V_DataGenerator import N2V_DataGenerator
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from n2v.models import N2VConfig, N2V
from tifffile import imwrite

from utils import reshape_2d, redirect_output, tif_to_matrix


def memory_alloc(n_gb: int):
    """
    Allocate memory on the GPU. If not activated the code will
    by default fill the GPU, without any performance gain.
    """
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [
                    tf.config.experimental.VirtualDeviceConfiguration(
                        memory_limit=1024 * n_gb
                    )
                ],
            )
        except RuntimeError as e:
            print(e)


def compute_scores(
    data: np.ndarray,
    f_dir: os.PathLike,
    d_name: str,
    save: bool = True
):
    f_dir = pathlib.Path(f_dir)
    f_dir.mkdir(parents=True, exist_ok=True)
    data = reshape_2d(data)

    # We now scale our data and compute the scores.
    scale = StandardScaler()
    data = scale.fit_transform(data)
    pca = PCA()
    scores = pca.fit_transform(data)
    if save:
        np.save(
            f_dir / f"{d_name}_scores.npy",
            scores.astype("float16")
        )
        np.save(
            f_dir / f"{d_name}_components.npy",
            pca.components_.astype(np.float16),
        )
    # We reshape the scores.
    return scores


def denoise_scores(
    j: int,
    scores_j: np.ndarray,
    batch_size: int,
    basedir: os.PathLike,
    model_name: str,
    patch_shape: Optional[Tuple[int, ...]] = None,
    train_epochs: int = 100,
    split_val: float = 0.9,
    hide_output: bool = False,
):
    basedir = pathlib.Path(basedir)
    dir_model = basedir / model_name
    dir_history = dir_model / "history"
    dir_denoised = dir_model / "denoised_scores"
    dir_model.mkdir(exist_ok=True)
    dir_denoised.mkdir(exist_ok=True)
    dir_history.mkdir(exist_ok=True)

    dat_j = data_augment(scores_j)
    dat_j = [np.expand_dims(dat, 0) for dat in dat_j]
    dat_j = [np.expand_dims(dat, -1) for dat in dat_j]

    if patch_shape is None:
        if scores_j.ndim == 2:
            patch_shape = (64, 64)
            axes = "YX"
        if scores_j.ndim == 3:
            patch_shape = (16, 16, 16)
            axes = "ZYX"

    # Generate patches
    datagen = N2V_DataGenerator()
    with redirect_output(hide_output):
        patches = datagen.generate_patches_from_list(dat_j, shape=patch_shape)

    np.random.shuffle(patches)
    patches = patches.astype(np.float32)  # float16 generates errors
    train_break = int(patches.shape[0] * split_val)
    X = patches[:train_break]
    X_val = patches[train_break:]

    # Configuration of N2V
    config = N2VConfig(
        X,
        unet_kern_size=3,
        train_steps_per_epoch=max(int(X.shape[0] / batch_size), 1),
        train_epochs=train_epochs,
        train_loss="mse",
        batch_norm=True,
        train_batch_size=batch_size,
        n2v_perc_pix=0.198,
        n2v_patch_shape=patch_shape,
        n2v_manipulator="uniform_withCP",
        n2v_neighborhood_radius=5,
        single_net_per_channel=False,
    )

    # We are now creating our network model.
    model = N2V(config, model_name, basedir=str(basedir))

    # Start training. The model saves itself automatically.
    with redirect_output(hide_output):
        history = model.train(X, X_val)

    # Save history and denoised images
    history = pd.DataFrame.from_dict(history.history)
    history.to_csv(dir_history / f"history_{j:04}.csv")

    if len(patch_shape) == 2:
        axes = "YX"
    elif len(patch_shape) == 3:
        axes = "ZYX"

    with redirect_output(hide_output):
        score_dn = model.predict(scores_j, axes)
    imwrite(dir_denoised / f"score_{j:04}.tiff", score_dn)


def reconstruct(
    data: np.ndarray,
    dir_scores_dn: os.PathLike,
    f_components: os.PathLike,
) -> np.ndarray:
    """Reconstruct the dataset into a 2d matrix of dimensions (n_pix, n_mz)"""

    data = data.astype(np.uint32)
    data = reshape_2d(data)
    means = np.mean(data, axis=1)
    std = np.std(data, axis=1)

    # Load components and scores
    components = np.load(f_components)
    scores_dn = tif_to_matrix(dir_scores_dn, as_sparse=False)
    # scores_dn = np.moveaxis(scores_dn, 0, 2)
    scores_dn = reshape_2d(scores_dn)

    # Reconstruct denoised data
    data_dn = scores_dn @ components
    data_dn = (data_dn.T * std + means).T
    data_dn[data_dn < 0] = 0

    return data_dn


def data_augment(data: np.ndarray):
    # This function augments the data by performing flips and transpositions.
    # print("Augmenting data...")
    if data.ndim == 2:  # data times 8
        res = [np.rot90(data, k=i) for i in range(4)]
        for dat in [np.rot90(data, k=i) for i in range(4)]:
            res.append(np.flip(dat, 0))
    if data.ndim == 3:
        res = [np.rot90(data, k=i, axes=(0, 1)) for i in range(4)] + [
            np.rot90(data, k=i, axes=(0, 2)) for i in range(4)
        ]
        res2 = []
        for dat in res:
            res2 = res2 + [np.flip(dat, 0)]
        res = res + res2
        res2 = [np.rot90(data, k=i, axes=(1, 2)) for i in range(4)]
        for dat in res2:
            res = res + [np.fliplr(dat)]
        res = res + res2
    # print(f"Data was augmented by a factor {len(res)}.")
    return res
