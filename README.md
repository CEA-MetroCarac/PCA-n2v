# PCA-n2v

This repository contains the implementation of `PCA-n2v` (Noise2Void), based on the package [n2v](https://github.com/juglab/n2v).

## Running the example notebook

This code should run on recent versions of Python, though the package `n2v` cannot be installed on 3.12+.

From a clean virtual or conda environment install the requirements:

```bash
pip install -r requirements.txt
```

If this fails due to missing `git`, install a copy of git from [here](https://git-scm.com/downloads).
We must install `n2v` from GitHub in order to have certain bugfixes which are not yet released to `PyPi`.

If you have a CUDA-capable GPU then you can also run:

```bash
pip install -r "tensorflow[and-cuda]<2.16"
```

though it would be best to follow the `n2v` readme for configuring Tensorflow and CUDA appropriately.

From this folder launch the Jupyter notebook server:

```
jupyter notebook 
```

then load the file `Example_denoising.ipynb`.

## Example data

The example data for the repository are available at the following [Zenodo page](https://zenodo.org/records/14761145). We use the file `Alga_raw.npz` placed in the `./data` directory.

## Note on data import

Appropriate data import is necessary before running PCA-n2v. MSI data should than be transformed into a matrix (.npy) or better a sparse matrix (.npz) before further processing. Keep in mind that the format `float16` is not supported in .npz matrixes.

For IonTOF systems, we recommend exporting all the data in `.txt` format. This creates as many files as are m/z. The function `iontof_to_matrix` has been especially implemented for this purpose.

For PHI data, we recommend exporting the data as `.tif` images in a dedicated folder. Warning: this functions only if the maximum value of pixels is 256, otherwise you might need to split one peak into several peaks. The function `tif_to_matrix` performs the task of turning a set of `.tif` images into a matrix.

We have not implemented functions for other systems but we will be very happy to include the functions you build for your own data into this Github as a service to the community.
