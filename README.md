# Bachelor Thesis Joscha Lasse Bisping

This repository contains the essential code from my bachelor thesis.


## Data

The already preprocessed data can be accessed via the following link:

https://drive.google.com/drive/folders/1io3YIOxBweMbQQMM5UtqYP7su1pTc25v?usp=sharing

For the experiment I used the `DATA_EEG/EEG_PREP_ROBUST_TRAIN` for training,
the `DATA_EEG/TEST_50` for testing and the `DATA_EEG/VAL_70` for validation.

These Folders `EEG_PREP_ROBUST_TRAIN`, `TEST_50` and `VAL_70` can be downloaded and placed in the `DATA_EEG` folder.

## Installation
During this project I used a singularity container to manage the environment and execution of all tests.

A quick guide how I set up the singularity container can be found here:

https://jlb-jlb.notion.site/Singularity-4-0-2-f15058d2e4fd486388f36f45ba80c60a


In case some dependencies are missing, you can add those to the Singularity definition file or you can install those later. If you choose to install them later they will not be written to the container, but rather a local directory which the container can accesss.

The .def file is available in this repository.
The .sif file can also be found and downloaded in the data folder.

## Experiment

For both the Experiment and Preprocessing, please ensure that the data paths match!

to run the experiment I suggest you start the container with 

```
singularity exec --nv jlbPyTorch.sif bash
```

and afterwards run one of the shell files inside the container. Example:

```
./Exp_0133_run_Autoformer.sh
```

## Preprocessing
If you want to preprocess the data on your own, you need to download it from the official Temple University Data Corpus. This requires a registration. The data is available under https://isip.piconepress.com/projects/tuh_eeg/

---

### Disclaimers

- The docstring of functions has been made with the help of ChatGPT.
- The code was formatted using black. <a href="#12">[12]</a>





### Code References

<a id="1"> [1] </a>
Alexandre Gramfort, Martin Luessi, Eric Larson, Denis A. Engemann, Daniel Strohmeier, Christian Brodbeck, Roman Goj, Mainak Jas, Teon Brooks, Lauri Parkkonen, and Matti S. Hämäläinen. MEG and EEG data analysis with MNE-Python. Frontiers in Neuroscience, 7(267):1–13, 2013. doi:10.3389/fnins.2013.00267 

<a id="2"> [2] </a>
Appelhoff, S., Hurst, A. J., Lawrence, A., Li, A., Mantilla Ramos, Y. J., O'Reilly, C., Xiang, L., Dancker, J., Scheltienne, M., Bialas, O., & Alibou, N. PyPREP: A Python implementation of the preprocessing pipeline (PREP) for EEG data. [Computer software]. https://github.com/sappelhoff/pyprep

<a id="3"> [3] </a>
Falcon, W., & The PyTorch Lightning team. (2019). PyTorch Lightning (Version 1.4) [Computer software]. https://doi.org/10.5281/zenodo.3828935

<a id="4"> [4] </a>
Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Kopf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., Bai, J., & Chintala, S. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library [Conference paper]. Advances in Neural Information Processing Systems 32, 8024–8035. http://papers.neurips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library.pdf

<a id="5"> [5] </a>
Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., Moi, A., Cistac, P., Ma, C., Jernite, Y., Plu, J., Xu, C., Le Scao, T., Gugger, S., Drame, M., Lhoest, Q., & Rush, A. M. (2020). Transformers: State-of-the-Art Natural Language Processing [Conference paper]. 38–45. https://www.aclweb.org/anthology/2020.emnlp-demos.6



<a id="6"> [6] </a>
The pandas development team. pandas-dev/pandas: Pandas [Computer software]. https://doi.org/10.5281/zenodo.3509134

<a id="7"> [7] </a>
Harris, C.R., Millman, K.J., van der Walt, S.J. et al. Array programming with NumPy. Nature 585, 357–362 (2020). https://doi.org/10.1038/s41586-020-2649-2


<a id="8"> [8] </a>
Ignacio Oguiza (2023): tsai. A state-of-the-art deep learning library for time series and sequential data: Github. Available online at https://github.com/timeseriesAI/tsai, checked on 1/11/2024.

<a id="9"> [9] </a>
Pedregosa, F. and Varoquaux, G. and Gramfort, A. and Michel, V. (2011): Scikit-learn: Machine Learning in {P}ython. In Journal of Machine Learning Research 12, 2825--2830.

<a id="10"> [10] </a>
Hunter, John D. (2007): Matplotlib: A 2D Graphics Environment. In Comput. Sci. Eng. 9 (3), pp. 90–95. DOI: 10.1109/MCSE.2007.55.

<a id="11"> [11] </a>
Detlefsen, Nicki; Borovec, Jiri; Schock, Justus; Jha, Ananya; Koker, Teddy; Di Liello, Luca et al. (2022): TorchMetrics - Measuring Reproducibility in PyTorch. In JOSS 7 (70), p. 4101. DOI: 10.21105/joss.04101.

<a id="12"> [12] </a>
Langa, Ł., & contributors to Black. Black: The uncompromising Python code formatter [Computer software]. https://github.com/psf/black