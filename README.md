reg_deconProject
=============

A code collection of the registration and deconvolution project corresponding to the paper: 

 Min Guo, *et al*. "[Rapid image deconvolution and multiview fusion for optical microscopy](https://doi.org/10.1038/s41587-020-0560-x)." Nature Biotechnology 38.11 (2020): 1337-1346.

This repository covers most functions and implementations reported in the paper. All codes, except the deep learning module `DenseDeconNet`, now run in MATLAB. Users may refer to the [code description](./Code_description.pdf) for more details and use the repository under the [license agreement](./LICENSE.pdf).  

The dependency C++/CUDA library for the codes has been compiled and attached in the release package. User can also find the source code of the C++/CUDA library at repository [**microImageLib**](https://github.com/eguomin/microImageLib).
In addition, the diSPIM data processing programs have been split to another repository [**diSPIMFusion**](https://github.com/eguomin/diSPIMFusion) and will be maintained independently.
