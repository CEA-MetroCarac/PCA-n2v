import os
import contextlib
import pathlib
from typing import Tuple, Optional
from typing_extensions import Literal
import numpy as np
from scipy import sparse
import pandas as pd
from tifffile import imwrite, imread


def reshape_2d(data: np.ndarray):
    """Reshape a 3d of 4d matrix into a 2d matrix."""
    return data.reshape(-1, data.shape[-1])


def tif_to_matrix(dir_tif: os.PathLike, as_sparse: bool = True):
    """
    Loads a dataset saved as a tif or tiff files in
    dir_tif and transforms it to a npy or sparse matrix.
    """
    dir_tif = pathlib.Path(dir_tif)
    list_im = [
        f for f in dir_tif.iterdir()
        if f.suffix in (".tif", ".tiff")
    ]
    data = [
        imread(f_name)
        for f_name
        in sorted(list_im, key=lambda f: f.stem)
    ]
    data = np.array(data)
    data = np.moveaxis(data, 0, -1)
    if as_sparse:
        return sparse.csr_matrix(reshape_2d(data))
    return data


def npz_to_tif(data_npz: sparse.csr_matrix, shape: Tuple[int, ...], dir_out: os.PathLike):
    """
    Saves the elements from a npz matrix into an output
    directory (dir_out) as a tiff file.
    The last dimension of the matrix is assumed to be the channel axis.
    """
    dir_out = pathlib.Path(dir_out)
    data = data_npz.todense()
    data = np.array(data).reshape(shape)
    data = np.moveaxis(data, -1, 0)
    for i, im in enumerate(data):
        imwrite(dir_out / f"{i:04}.tiff", im)


def iontof_to_matrix(
    dir_csv: os.PathLike,
    dir_out: os.PathLike,
    d_type: Literal["2d", "3d"] = "2d",
    extension: Optional[Literal["npz", "npy"]] = "npz",
):
    """
    Takes as input a file where IonTof data has been saved as txt file.
    Returns and saves the data in the file as a npy or npz file.
    """
    data_full = []
    for f_name in pathlib.Path(dir_csv).iterdir():
        if f_name.suffix == ".txt":
            data_mz, shape = txt_to_matrix(f_name, d_type)
            data_full.append(data_mz)
    if len(data_full) == 0:
        raise RuntimeError("No .txt data found")
    data_full = np.array(data_full)
    data_full = np.moveaxis(data_full, 0, -1)
    dir_out = pathlib.Path(dir_out)
    if extension is not None:
        if extension == "npz":
            sparse.save_npz(
                dir_out / "data.npz",
                sparse.csr_matrix(data_full)
            )
        elif extension == "npy":
            np.save(
                dir_out / "data.npy",
                data_full
            )
        else:
            raise ValueError(f"Unrecognized extension {extension}")
    return data_full


def txt_to_matrix(
    f_csv: os.PathLike,
    d_type: Literal["2d", "3d"] = "2d",
    dir_save: Optional[os.PathLike] = None,
):
    """
    Transforms a txt file exported from SurfaceLab into a matrix.
    If dir_save, also saves it as a .npy matrix."""
    
    if d_type == "2d":
        names = ["x", "y", "i"]
        dtype = {"x": int, "y": int, "i": float}
    elif d_type == "3d":
        names = ["x", "y", "z", "i"]
        dtype = {"x": int, "y": int, "z": int, "i": float}
    else:
        raise ValueError(f"Data type not supported {d_type}")

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
        shape = (np.max(x) + 1, np.max(y) + 1)
        data_temp = np.empty(shape = shape)
        data[x, y] = i
    elif d_type == "3d":
        z = data_mz["z"].to_numpy()
        shape = (np.max(x) + 1, np.max(y) + 1, np.max(z) + 1)
        data_temp = np.empty(shape = shape)
        data_temp[x, y, z] = i
        
    if dir_save is not None:
        filename = pathlib.Path(f_csv).stem
        sparse.save_npz(
            pathlib.Path(dir_save) / f"{filename}.npy",
            data_temp
        )
    return data_temp, shape


def load_data(f_name: os.PathLike):
    """
    Loads a MSI dataset saved as a .npz sparse matrix.
    """
    data = sparse.load_npz(f_name)
    return np.array(data.todense())


@contextlib.contextmanager
def redirect_output(redirect: bool):
    if redirect:
        # Redirect messages to devnull
        with open(os.devnull, "w") as dn:
            with contextlib.redirect_stderr(dn), contextlib.redirect_stdout(dn):
                yield
    else:
        yield
