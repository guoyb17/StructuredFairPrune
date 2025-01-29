# StructuredFairPrune: Fairness-aware Structured Pruning

Repo for paper **Fairness-aware Structured Pruning of Deep Learning Models** (submitted to TCAD).

## Update

### 1/29/2025

Initial commit.
Updates to this repository will be listed here.

## Dependency

The repo has been built and run on Python 3.9.18.

First, install PyTorch and related packages. See [PyTorch website](https://pytorch.org/) for installation instructions that work for your machine.

Then, install the following packages (the command is for conda; modify according to your environment):

```
conda install pandas bokeh tqdm scikit-learn scikit-image
```

```
pip install torch-pruning
```

Also, install `inq` from [INQ-pytorch](https://github.com/Mxbonn/INQ-pytorch), which offers its own installation methods.

## Dataset

The datasets are too large, so we do not upload it to GitHub, but we offer links for you to look for or simly download them.

### Fitzpatrick 17k

**Note**: so far, our paper has not used Fitzpatrick 17k for any experiment in this paper yet, but it has been used in the original FairPrune paper as well as many of our other research.

[Offical website](https://github.com/mattgroh/fitzpatrick17k)

[Packaged dataset powered by Notre Dame Box](https://notredame.box.com/s/pjf9kw5y1rtljnh81kuri4poecuiqngf)

[Direct link powered by Notre Dame Box](https://notredame.box.com/shared/static/pjf9kw5y1rtljnh81kuri4poecuiqngf.tar)

The following commands will make the data ready for default paths in the code:

```
wget https://notredame.box.com/shared/static/pjf9kw5y1rtljnh81kuri4poecuiqngf.tar -O fitzpatrick17k.tar
tar –xvf fitzpatrick17k.tar
```

### ISIC 2019

[Offical website](https://challenge.isic-archive.com/landing/2019/)

[Packaged dataset powered by Notre Dame Box](https://notredame.box.com/s/uw8g5urs7m4n4ztxfo100kkga6arzi9k)

[Direct link powered by Notre Dame Box](https://notredame.box.com/shared/static/uw8g5urs7m4n4ztxfo100kkga6arzi9k.tar)

The following commands will make the data ready for default paths in the code:

```
wget https://notredame.box.com/shared/static/uw8g5urs7m4n4ztxfo100kkga6arzi9k.tar -O ISIC_2019_train.tar
tar –xvf ISIC_2019_train.tar
```

### CelebA

[Offical website](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

[Packaged dataset powered by Notre Dame Box](https://notredame.box.com/s/s2ais65ldhzpm6wx4sej11ltecajbtqt)

[Direct link powered by Notre Dame Box](https://notredame.box.com/shared/static/s2ais65ldhzpm6wx4sej11ltecajbtqt.tar)

The following commands will make the data ready for default paths in the code:

```
wget https://notredame.box.com/shared/static/s2ais65ldhzpm6wx4sej11ltecajbtqt.tar -O img_align_celeba.tar
tar –xvf img_align_celeba.tar
```

## Usage

For most modules (e.g., `pre_train.py`), use `--help` or `-h` for usage information.

```
python pre_trained.py --help
```

There are many paths to files and directories that are hard-coded or default in the codes.
You may download the packaged datasets from Notre Dame Box links provided above and decompress to match the default paths.
Alternatively, you may also configure by yourself.

`pre_trained.py` is to train a vanilla model on the selected dataset.

`pruning_examples/select_examples.py` should be run once before all pruning experiments.
It randomly picks pruning examples required by the pruning framework from the designated datset(s).
Therefore, please run it after you get datsets ready.

`prune.py` applies structured pruning to a given model.

`test.py` and `test_group.py` test given models.
`test.py` tests one model per time, while `test_group.py` tests all models in a given directory.
If you have a bunch of models to test (e.g., output models from pruning), `test_group.py` would be faster than calling `test.py` for each model.

`test_deploy.py` tests hardware efficiency performance on a given environment.
It does not test accuracy or fairness.

`test_deploy.py` uses `--device` to determine which device to run if multiple options are available.
Now support: `cpu` for CPUs, `cuda` for NVIDIA GPUs with CUDA support, `mps` for Apple Silicon GPUs with MPS support.
Other codes automatically determine the device to run based on availability with the priority order of CUDA, MPS, and CPU (from high to low).

## Contact

If you have any question, feel free to submit issues and/or email me (Yuanbo Guo): yguo6 AT nd DOT edu.
Thank you so much for your support!
