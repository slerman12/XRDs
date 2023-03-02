# Data

First, download and generate the XRD data as described in the Readme [here](Datasets/Generated).

# Installation

This project is built with the [UnifiedML](https://github.com/AGI-init/UnifiedML) deep learning library/framework.

First, clone this current project:

```console
git clone git@github.com:agi-init/XRD.git
cd XRD
```

Next, install UnifiedML:

1. Clone repo:

```console
git clone git@github.com:agi-init/UnifiedML.git
```

2. Install dependencies via [Conda](https://docs.conda.io/en/latest/miniconda.html):

```console
conda env create --name ML --file=Conda.yml
```

3. Activate:

```console
conda activate ML
```

# Running

To run, we have 3 model variants:

```console
# No-pool CNN model
python XRD.py task=NPCNN

# Standard CNN model
python XRD.py task=SCNN

# MLP model
python XRD.py task=MLP
```

Which can be used to predict 7-way crystal types. To predict 230-way space groups instead, add the ```num_classes=230``` flag. For example,

```console
# No-pool CNN model - predicting 230-way space groups
python XRD.py task=NPCNN num_classes=230
```

This will launch training on the Soup data (synthetic + random 50% RRUFF), & evaluation on the remaining 50% RRUFF data. The trained model is saved in a ```./Checkpoints``` directory and can be loaded with the ```load=true``` flag.

All [UnifiedML](https://github.com/AGI-init/UnifiedML) features and syntax are supported.

# Paper

If you find this work useful, be sure to cite us:

```bibtex
@article{crystallographic~2023,
  title   = {Classifying crystals with deep learning at scale},
  author  = {Salgado, Jerardo ...},
  journal = {Nature Communications pre-print},
  year    = {2023}
}
```