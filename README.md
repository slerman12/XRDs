![alt text](https://i.imgur.com/ex7bPB0.png)

**This is the repo for classifying crystal structures & space groups via 1D X-ray diffraction patterns (XRDs).**

> &#9432; *XRDs are the incident shadows of light beams struck through materials like iron or copper, which are composed of crystal lattices that identify the unique materials and its properties. A rotating X-ray shoots photons or electrons at a crystal and leaves an incident marker that is integrated across the firing axis into a 1D data representation. These shadows are fast to create but hard to reverse engineer to their original crystallographic nature. In this work, we use machine learning to predict the material's type and space group from these simple 1D patterns, with accuracies attained above 80%, automating a difficult laborious task that is traditionally done by human hand.*

**[Check out our paper for more details and information, and be sure to cite us.]()**
 
```bibtex
@article{crystallographic~2023,
title   = {Classifying crystals with deep learning at scale},
author  = {Salgado, Jerardo ...},
journal = {Nature Communications pre-print},
year    = {2023}
}
```

---

## Data

**Download and generate the 1D XRD data as described in the Readme [here](Datasets/Generated).**

## Installation

### 1. Clone Current Project

```console
git clone git@github.com:agi-init/XRD.git
cd XRD
```

### 2. Install UnifiedML

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

---

The above scripts will launch training on the Soup data (synthetic + random 50% RRUFF), & evaluation on the remaining 50% RRUFF data. The trained model is saved in a ```./Checkpoints``` directory and can be loaded with the ```load=true``` flag.

Plots automatically save to ```./Benchmarking/<experiment>/```
:chart_with_upwards_trend: :bar_chart: --> ```./Benchmarking/Exp/```

**All [UnifiedML](https://github.com/AGI-init/UnifiedML) features and syntax are supported.**

---

## Citing

If you find this work useful, be sure to cite us:

```bibtex
@article{crystallographic~2023,
  title   = {Classifying crystals with deep learning at scale},
  author  = {Salgado, Jerardo ...},
  journal = {Nature Communications pre-print},
  year    = {2023}
}
```

---

[This code is licensed under the MIT license.](MIT_LICENSE)