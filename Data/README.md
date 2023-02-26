# Synthetic XRD Data Generation

Code generates 1D XRD patterns using information extracted from CIF files.

## Description

Because experimental XRD data is difficult to obtain, we have developed our own python code to generate synthetic patterns
from CIF files. Notebook 1 generates the XRD patterns using CIF files as input. The code calls to a folder containing all
CIF files and calculates the patterns sequentially. Notebook 1 formats the data to be used for model development.

## Getting Started

### Dependencies

* Any relatively recent hardware can run the code

### Installing

* Input and output directories are defined and created within the code
* A separate CIF folder was made using Materials Project CIF files (~2500)

### Executing program

* Running the program requires 2 notebook executions, generating the XRD patterns and formatting for model training
* Step 1: execute XRD_1D_pipeline_using_CIFs.ipynb
* Step 2: execute XRD_1D_pipeline_converter_for_training.ipynb


## Authors

Contributors names and contact info

Zhaotong Du  
zdu3@ur.rochester.edu

Jerardo Salgado
jsalgad2@ur.rochester.edu

## Version History

* 0.1
    * Initial Release
