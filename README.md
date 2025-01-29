# PCA-n2v

This repository contains


## Note on data importation

Data importation is the first step to performing PCA-n2v. MSI data should than be transformed into a matrix (.npy) or better a sparse matrix (.npz) before further processing. Keep in mind that the format 'float16' is not supported in .npz matrixes.

For IonTOF systems, we recommend exporting all the data in .txt format. This creates as many files as are m/z. The function "iontof_to_matrix" has been especially implemented for this purpose.

For PHI data, we recommend exporting the data as .tif images in a dedicated folder. Warning: this functions only if the maximum value of pixels is 256, otherwise you might need to split one peak into several peaks. The function "tif_to_matrix" performs the task of turning a set of .tif images into a matrix.

We have not implemented functions for other systems but we will be very happy to include the functions you build for your own data into this Github as a service to the community.
