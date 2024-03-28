# Bachelor Thesis Joscha Lasse Bisping

This repository contains the essential code from my bachelor thesis.


# Data

The already preprocessed data can be accessed via the following link:

https://drive.google.com/drive/folders/1io3YIOxBweMbQQMM5UtqYP7su1pTc25v?usp=sharing

For the experiment I used the `DATA_EEG/EEG_PREP_ROBUST_TRAIN` for training,
the `DATA_EEG/TEST_50` for testing and the `DATA_EEG/VAL_70` for validation.

These Folders `EEG_PREP_ROBUST_TRAIN`, `TEST_50` and `VAL_70` can be downloaded and placed in the `DATA_EEG` folder.

# Installation
During this project I used a singularity container to manage the environment and execution of all tests.

A quick guide how I set up the singularity container can be found here:

https://jlb-jlb.notion.site/Singularity-4-0-2-f15058d2e4fd486388f36f45ba80c60a?pvs=4


In case some dependencies are missing, you can add those to the Singularity definition file or you can install those later. If you choose to install them later they will not be written to the container, but rather a local directory which the container can accesss.


## Experiment

to run the experiment I suggest you start the container with 

```
singularity exec --nv jlbPyTorch.sif bash
```

and afterwards run one of the shell files inside the container. Example:

```
./Exp_0133_run_Autoformer.sh
```



# credits

tsai

gluonts
