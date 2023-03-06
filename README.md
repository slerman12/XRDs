![alt text](https://i.imgur.com/0Qp4YOb.png)

**Welcome to classifying crystal structures & space groups from 1D X-ray diffraction (XRD) patterns.**

*Can machine learning identify crystals in lossy light diffraction patterns?* </br>
**[Check out our paper for more details and information, and be sure to cite us.]()**
 
---

Paper:

```bibtex
@article{Crystals,
title   = {XRDs with deep learning (pending actual name)},
author  = {Jerardo Salgado; Sam Lerman; Zhaotong Du; Chenliang Xu; and Niaz Abdolrahim},
journal = {pre-print:Nature Communications},
year    = {2023}
}
```

---

# :point_up: Setup

**Download and generate the 1D XRD data as described [in the Readme here](Datasets/Generated).**

## 1. Clone Current Project

Use **[git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)** to download the XRDs repo:

```console
git clone git@github.com:slerman12/XRDs.git
```

Change directory into the XRDs repo:

```console
cd XRDs
```

## 2. Install UnifiedML

This project is built with the **[UnifiedML](https://github.com/AGI-init/UnifiedML)** deep learning library/framework.

**Download UnifiedML**

```console
git clone git@github.com:agi-init/UnifiedML.git
```

**Install Dependencies**

All dependencies can be installed via **[Conda](https://docs.conda.io/en/latest/miniconda.html)**:

```console
conda env create --name ML --file=UnifiedML/Conda.yml
```

**Activate Conda Environment**

```console
conda activate ML
```

#

> &#9432; If your GPU doesn't support the latest CUDA version, you may need to redundantly install Pytorch with an older version of CUDA from [pytorch.org/get-started](https://pytorch.org/get-started/locally/) after activating your Conda environment. For example, for CUDA 11.6:
> ```console
> pip uninstall torch torchvision torchaudio
> pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
> ```
> &#9432; CUDA is needed to run the deep learning code on GPUs rather than CPUs. UnifiedML will automatically select GPUs when a working CUDA is available.

[//]: # (${\text{\color{green}✓}}$ **Done** )

---

# Reproducing The Work

Once data is downloaded and UnifiedML is installed, reproducing is easy.

[//]: # (<summary><h1 style="display: inline-block;">Reproducing paper</h1></summary>)

To run, we have 3 model variants for predicting **7-way crystal types**:

**Model 1: No-pool CNN**

```console
python XRD.py task=NPCNN
```

**Model 2: Standard CNN**

```console
python XRD.py task=SCNN
```

**Model 3: MLP**

```console
python XRD.py task=MLP
```

:bulb: **To predict 230-way space groups instead**, add the ```num_classes=230``` flag.

```console
python XRD.py task=NPCNN num_classes=230
```

Plots automatically save to ```./Benchmarking/<experiment>/```.

#

The above scripts will launch training on the "souped" **synthetic + random 50% RRUFF experimental data**, & evaluation on the **remaining 50% RRUFF data**. The trained model is saved in a ```./Checkpoints``` directory and can be loaded with the ```load=true``` flag.

**All model code can be found in [```XRD.py```](XRD.py)**

---

# Citing

If you find this work useful, be sure to cite us:

```bibtex
@article{Crystals,
title   = {XRDs with deep learning (pending actual name)},
author  = {Jerardo Salgado; Sam Lerman; Zhaotong Du; Chenliang Xu; and Niaz Abdolrahim},
journal = {pre-print:Nature Communications},
year    = {2023}
}
```

---

All **[UnifiedML](https://github.com/AGI-init/UnifiedML)** features and syntax are supported.

#

[This code is licensed under the MIT license.](MIT_LICENSE)