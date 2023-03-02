![alt text](https://i.imgur.com/0Qp4YOb.png)

**This is the repo for classifying crystal structures & space groups via 1D X-ray diffraction patterns (XRDs).**

*Can machine learning be used to deduce the structure of the original material?* </br>
**[Check out our paper for more details and information, and be sure to cite us.]()**
 
---

```bibtex
@article{Crystals,
title   = {XRDs with deep learning},
author  = {Jerardo Salgado; Sam Lerman; Zhaotong Du; Chenliang Xu; and Niaz Abdolrahim},
journal = {pre-print:Nature Communications},
year    = {2023}
}
```

---

# :point_up: Setup

**Download and generate the 1D XRD data as described [in the Readme here](Datasets/Generated).**

## 1. Clone Current Project

Use [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) to download the XRDs repo:

```console
git clone git@github.com:slerman12/XRDs.git
```

Change directory into the XRDs repo:

```console
cd XRDs
```

## 2. Install UnifiedML

This project is built with the [UnifiedML](https://github.com/AGI-init/UnifiedML) deep learning library/framework.

**Download UnifiedML**

```console
git clone git@github.com:agi-init/UnifiedML.git
```

**Install Dependencies**

All dependencies installed via [Conda](https://docs.conda.io/en/latest/miniconda.html):

```console
conda env create --name ML --file=UnifiedML/Conda.yml
```

**Activate Conda Environment**

```console
conda activate ML
```

#

> &#9432; Depending on your CUDA version, you may need to redundantly uninstall and reinstall Pytorch with CUDA from [pytorch.org/get-started](https://pytorch.org/get-started/locally/) after activating your Conda environment. For example, for CUDA 11.6:
> ```console
> pip uninstall torch torchvision torchaudio
> pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
> ```

---

## Reproducing Paper

To run, we have 3 model variants for predicting **7-way crystal types**:

**No-pool CNN model**

```console
python XRD.py task=NPCNN
```

**Standard CNN model**

```console
python XRD.py task=SCNN
```

**MLP model**

```console
python XRD.py task=MLP
```

:bulb: **To predict 230-way space groups instead**, add the ```num_classes=230``` flag.

```console
python XRD.py task=NPCNN num_classes=230
```

Plots automatically save to ```./Benchmarking/<experiment>/```

---

The above scripts will launch training on the "souped" **synthetic + random 50% RRUFF data**, & evaluation on the **remaining 50% RRUFF data**. The trained model is saved in a ```./Checkpoints``` directory and can be loaded with the ```load=true``` flag.

---

## Paper & Citing

If you find this work useful, be sure to cite us:

```bibtex
@article{crystallographic~2023,
  title   = {Classifying crystals with deep learning at scale},
  author  = {Salgado, Jerardo ...},
  journal = {Nature Communications pre-print},
  year    = {2023}
}
```

#

All [UnifiedML](https://github.com/AGI-init/UnifiedML) features and syntax are supported.

#

[This code is licensed under the MIT license.](MIT_LICENSE)