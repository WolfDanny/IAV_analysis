[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
# Statistical analysis of IAV infection data from mice

This repository contains:
* Python module used to analyse flow-cytometry data from IAV infected mice (`flowanalysis/definitions.py`).
* Population frequencies, and numbers obtained from flow-cytometry (`Data/`).
* IPython notebook with the analysis of the data (`Figures.ipynb`)
* R markdown document for the analysis of variance (ANOVA) of the data (`ANOVA.rmd`).
* Python code to plot the results of the ANOVA using LaTeX (`ANOVA-networks.py`).
* Environment file to create the Anaconda environment used (`environment.yml`).

## Creating the Anaconda environment

To create and activate the Anaconda environment used run the following commands from the current directory:
```bash
conda env create -f environment.yml
conda activate IAV-analysis
```
> Note that this environment includes both the python and R modules used in the codes.

## Using the codes

Once the environment has been activated:
1. Run all the cells in `Figures.ipynb`. This will calculate and plot correlations between populations, generate other figures, and generate the datasets used for ANOVA.
2. Run all the cells in `ANOVA.rmd`. This will perform the analysis of variance for the data and save the results to be plotted.
3. Run `ANOVA-networks.py`. This will generate the LaTeX code to plot the results of the ANOVA and Tukey's HSD.
