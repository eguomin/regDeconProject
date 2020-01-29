# reg_deconProject
A code collection of the registration and deconvolution project corresponding to the preprint paper at bioRxiv (Guo et al, Accelerating iterative deconvolution and multiview fusion by orders of magnitude): https://www.biorxiv.org/content/10.1101/647370v1.abstract

This repository covers most functions and implementations reported in the paper. All codes, except the deep learning program DenseDeconNet, now run in MATLAB. The dependency C++/CUDA library for the codes has been compiled and attached in the release package. User can also find the source code of the C++/CUDA library at: https://github.com/eguomin/microImageLib.

In addition, the diSPIM data processing programs have been split to another repository and will be maintained independently: https://github.com/eguomin/diSPIMFusion
