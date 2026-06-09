# MLACCA: Machine Learning-Assisted Compound Class Assignment

Here, code is presented to create and use supervised classification models for elemental ratios (O/C, H/C, and N/C)-based compound class assignment derived from mass spectrometry-derived molecular formula assignment.
All modules used can be found in the `files/requirements.txt` file.

All material is in the `files` folder. Of special interest may be the pickled models contained in `files/data/models` and the demos presented in `files/demos`.

## Demos
The demos are organised so that end users may benefit from this method (almost) regardless of their coding skill level:
- In the `files/demo/demo--jupyter` folder, the demo is aimed at those who want to integrate the classification models directly into their own Python-based MS data analysis pipeline.
- In the `files/demo/demo--command_line` folder, the demo explains how to use this method from the command line, thus requiring some familiarity with the command line interface but relatively little Python knowledge.
- Finally, in the `files/demo/demo--app` a Python-based GUI application is presented. Here, some IT knowledge is still valuable as the correct Python environment needs to be activated (as in the other cases as well), but the app itself guides the user through the process.

## Worflow to Generate and Benchmark Models
1. `files/create_csvs.ipynb`: collate all the separate CSV files containing the necessary data into one large dataset file, alongside generating figures summarising dataset properties.
2. `files/hyperparameters_optimisation.ipynb`: optimise the hyperparameters of the supervised classifiers (*k*NN, GNB, polynomial and RBF SVM) on the newly created dataset.
3. `files/benchmark.ipynb`: benchmark the models against established definitions.
    - `files/compare_ru.ipynb`: for further benchmarking with the [Rivas-Ubach *et al.* (2018)](10.1021/acs.analchem.8b00529) validation dataset.
5. `files/model_generation`: generate and serialise ("pickle") the MLACCA models.
6. `files/3d_plots.ipynb`: generate 3D plots of the decision boundaries.
