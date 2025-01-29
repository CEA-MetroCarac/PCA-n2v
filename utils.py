from n2v.models import N2VConfig, N2V
import tensorflow as tf
import numpy as np
from n2v.internals.N2V_DataGenerator import N2V_DataGenerator
from scipy import sparse
import os
from tifffile import imwrite, imread
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd


def memory_alloc(n_go):
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
                        memory_limit=1024 * n_go
                    )
                ],
            )
        except RuntimeError as e:
            print(e)


def reshape_2d(data):
    """Reshape a 3d of 4d matrix into a 2d matrix."""
    mz = data.shape[-1]
    n_pix = int(data.size / mz)
    data = data.reshape(n_pix, mz)
    return data


def tif_to_matrix(dir_tif, extension="npz"):
    """
    Loads a dataset saved as a tif or tiff files in
    dir_tif and transforms it to a npy or npz matrix.
    """
    list_im = os.listdir(dir_tif)
    list_im = np.sort(list_im)
    data = [
        imread(os.path.join(dir_tif, f_name))
        for f_name in list_im
        if ".tif" in f_name
    ]
    data = np.array(data)
    data = np.moveaxis(data, 0, -1)
    if extension == "npz":
        data = sparse.csr_matrix(reshape_2d(data))
    return data


def npz_to_tif(data_npz, shape, dir_out):
    """
    Saves the elements from a npz matrix into an output
    directory (dir_out) as a tiff file.
    The last dimension of the matrix is assumed to be the channel axis.
    """
    data = sparse.csr_matrix.todense(data_npz)
    data = np.array(data).reshape(shape)
    data = np.moveaxis(data, -1, 0)
    data = list(data)
    for i, im in enumerate(data):
        imwrite(os.path.join(dir_out, f"{i:04}.tiff"), im)


def iontof_to_matrix(dir_csv, dir_out, d_type="2d", extension="npz"):
    """Takes as input a file where IonTof data has been saved as txt file.
    Returns and saves the data in the file as a npy or npz file."""
    list_csv = os.listdir(dir_csv)
    # data = np.zeros(shape=())
    for f_name in list_csv:
        data_full = []
        if "txt" in f_name:
            data_mz, shape = txt_to_matrix(f_name, d_type)
            data_full.append(data_mz)
    # mz = len(data_full)
    data_full = np.array(data_full)
    data_full = np.moveaxis(data_full, 0, -1)
    if extension == "npz":
        sparse.save_npz(os.path.join(dir_out, "data.npz"), sparse.csr_matrix(data_full))
    elif extension == "npy":
        # shape = shape + (mz,)
        # data = data.reshape(shape)
        np.save(os.path.join(dir_out, "data.npy"), data_full)
    return data_full


def txt_to_matrix(f_csv, d_type="2d", dir_save=None):
    """
    Transforms a txt file exported from SurfaceLab into a matrix."""
    if d_type == "2d":
        names = ["x", "y", "i"]
        dtype = {"x": int, "y": int, "i": float}
    elif d_type == "3d":
        names = ["x", "y", "z", "i"]
        dtype = {"x": int, "y": int, "z": int, "i": float}
    else:
        print("Data type not supported")

    data_mz = pd.read_csv(
        f_csv,
        skiprows=10,
        delim_whitespace=True,
        header=None,
        names=names,
        dtype=dtype,
    )

    x = data_mz["x"].to_numpy()
    y = data_mz["y"].to_numpy()
    i = data_mz["i"].to_numpy().astype(int)

    if d_type == "2d":
        shape = (len(x), len(y))
        data_temp = np.zeros(shape=shape)
        data_temp[x, y] = i
    elif d_type == "3d":
        z = data_mz["z"].to_numpy()
        shape = (len(x), len(y), len(z))
        data_temp = np.zeros(shape=shape)
        data_temp[x, y, z] = i

    data_temp = data_temp.flatten()
    if dir_save is not None:
        sparse.save_npz(
            os.path.join(dir_save, f"{f_csv}.npy"),
            sparse.csr_matrix(data_temp)
        )
    return data_temp, shape


def load_data(f_name):
    """
    Loads a MSI dataset saved as a .npz sparse matrix.
    """
    data = sparse.load_npz(f_name)
    data = np.array(sparse.csr_matrix.todense(data))
    return data


def compute_scores(data, f_dir, d_name, save=True):
    print("Computing scores...")

    os.makedirs(f_dir, exist_ok=True)
    data = reshape_2d(data)

    # We now scale our data and compute the scores.
    scale = StandardScaler()
    data = scale.fit_transform(data)
    pca = PCA()
    scores = pca.fit_transform(data)
    if save:
        np.save(os.path.join(f_dir, f"{d_name}_scores.npy"), scores.astype("float16"))
        np.save(
            os.path.join(f_dir, f"{d_name}_components.npy"),
            pca.components_.astype("float16"),
        )
    # We reshape the scores.
    return scores


def denoise_scores(
    j,
    scores_j,
    batch_size,
    basedir,
    model_name,
    patch_shape=None,
    train_epochs=100,
    split_val=0.9,
):
    print(f"Denoising score {j}...")
    datagen = N2V_DataGenerator()
    dir_model = os.path.join(basedir, f"{model_name}")
    dir_history = os.path.join(dir_model, "history")
    dir_denoised = os.path.join(dir_model, "denoised_scores")
    os.makedirs(dir_model, exist_ok=True)
    os.makedirs(dir_denoised, exist_ok=True)
    os.makedirs(dir_history, exist_ok=True)

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
    patches = datagen.generate_patches_from_list(dat_j, shape=patch_shape)

    np.random.shuffle(patches)
    patches = patches.astype("float32")  # float16 generates errors
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
    model = N2V(config, model_name, basedir=basedir)

    # Start training. The model saves itself automatically.
    history = model.train(X, X_val)

    # Save history and denoised images
    history = pd.DataFrame.from_dict(history.history)
    history.to_csv(os.path.join(dir_history, f"history_{j:04}.csv"))

    if len(patch_shape) == 2:
        axes = "YX"
    elif len(patch_shape) == 3:
        axes = "ZYX"

    score_dn = model.predict(scores_j, axes)
    imwrite(os.path.join(dir_denoised, f"score_{j:04}.tiff"), score_dn)


def reconstruct(data, dir_scores_dn, f_components):
    """Reconstruct the dataset into a 2d matrix of dimensions (n_pix, n_mz)"""

    print("Reconstructing data...")

    data = data.astype("uint32")
    data = reshape_2d(data)
    means = np.mean(data, axis=1)
    std = np.std(data, axis=1)

    # Load components and scores
    components = np.load(f_components)
    scores_dn = tif_to_matrix(dir_scores_dn)

    scores_dn = np.moveaxis(np.array(scores_dn), 0, 2)
    scores_dn = reshape_2d(scores_dn)

    # Reconstruct denoised data
    data_dn = scores_dn @ components
    data_dn = (data_dn.T * std + means).T
    data_dn[data_dn < 0] = 0

    return data_dn


def data_augment(data):
    # This function augments the data by performing flips and transpositions.
    print("Augmenting data...")
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
    print(f"Data was augmented by a factor {len(res)}.")
    return res
