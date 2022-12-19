[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
# Statistical analysis of IAV infection data from mice

This repository contains:
* Python module used to analyse flow-cytometry data from IAV infected mice (`flowanalysis/definitions.py`).
* Flow-cytometry data analysed (`Data/`).
* IPython notebook with the analysis of the data (`Figures.ipynb`)
* Environment file to create the Anaconda environment used (`environment.yml`).

## Creating the Anaconda environment

To create and activate the Anaconda environment used run the following commands from the current directory:
```bash
conda env create -f environment.yml
conda activate IAV-analysis
```
