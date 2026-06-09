[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/release/python-31311/)

# MLACCA: Machine Learning-Assisted Compound Class Assignment

Here, code is presented to create and use supervised classification models for elemental ratios (O/C, H/C, and N/C)-based compound class assignment derived from mass spectrometry-derived molecular formula assignment.
All modules used can be found in the [`files/requirements.txt`](files/requirements.txt) file.

All material is in the [`files`](files/) folder. Of special interest may be the pickled models contained in [`files/data/models`](files/data/models) and the demos presented in [`files/demos`](files/demos).

## Demos
The demos are organised so that end users may benefit from this method (almost) regardless of their coding skill level:
- In the [`demo--jupyter`](files/demo/demo--jupyter) folder, the demo is aimed at those who want to integrate the classification models directly into their own Python-based MS data analysis pipeline.
- In the [`demo--command_line`](files/demo/demo--command_line) folder, the demo explains how to use this method from the command line, thus requiring some familiarity with the command line interface but relatively little Python knowledge.
- Finally, in the [`demo--app`](files/demo/demo--app) a Python-based GUI application is presented. Here, some IT knowledge is still valuable as the correct Python environment needs to be activated (as in the other cases as well), but the app itself guides the user through the process.

## Worflow to Generate and Benchmark Models
1. [`create_csvs.ipynb`](files/create_csvs.ipynb): collate all the separate CSV files containing the necessary data into one large dataset file, alongside generating figures summarising dataset properties.
2. [`hyperparameters_optimisation.ipynb`](files/hyperparameters_optimisation.ipynb): optimise the hyperparameters of the supervised classifiers (*k*NN, GNB, polynomial and RBF SVM) on the newly created dataset.
3. [`benchmark.ipynb`](files/benchmark.ipynb): benchmark the models against established definitions.
    - [`compare_ru.ipynb`](files/compare_ru.ipynb): for further benchmarking with the [Rivas-Ubach *et al.* (2018)](https://doi.org/10.1021/acs.analchem.8b00529) validation dataset.
5. [`model_generation.ipynb`](files/model_generation.ipynb): generate and serialise ("pickle") the MLACCA models.
6. [`3d_plots.ipynb`](files/3d_plots.ipynb): generate 3D plots of the decision boundaries.
