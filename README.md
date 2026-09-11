[![License: MPL 2.0](https://img.shields.io/badge/License-MPL%202.0-blue.svg)](https://opensource.org/licenses/MPL-2.0)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue?logo=python)](https://www.python.org/downloads/release/python-31311/)
[![Jupyter](https://img.shields.io/badge/Jupyter-blue?logo=jupyter)](https://jupyter.org/)

![SSEFCCA logo](logo/ssefcca_logo--black_background.svg)

# SSEFCCA: Supervised Semi-Empirical Formula Compound Class Assignment

Untargeted mass spectrometry can provide molecular formula assignments for thousands of detected analytes in complex mixtures; however, routine structural elucidation at this scale remains impractical.
SSEFCCA provides a principled, Python-based, and data‑driven approach to (i) improve atomic ratios (O/H, H/C, and N/C)-based compound class assignment accuracy and (ii) evaluate the assignment confidence, thus supporting more defensible and critical downstream comparisons of compoud class diversity across samples.

The trained classifiers can be found [pickled](https://docs.python.org/3/library/pickle.html) at [`files/data/models`](files/data/models). While we recommend using the polynomial SVM model, *k*NN, GNB, and RBF SVM classifiers were also trained and are available for use.

More broadly, SSEFCCA represents a blueprint for programmatic retraining and expansion to additional classes of supervised classifiers as the needs of the wider metabolomics community evolve.


## Demos
For more information on how to integrate SSEFCCA in your workflow with worked examples, you may want to check out the [`files/demos`](files/demos) folder.

The demos were made so that end users may benefit from this method (almost) regardless of their coding skill level:
- In the [`demo--jupyter`](files/demos/demo--jupyter) folder, the demo is aimed at those who want to integrate the classification models directly into their own Python-based MS data analysis pipeline.
- In the [`demo--command_line`](files/demos/demo--command_line) folder, the demo explains how to use this method from the command line, thus requiring some familiarity with the command line interface but relatively little Python knowledge.
- Finally, in the [`demo--app`](files/demos/demo--app) a Python-based GUI application is presented. Here, some IT knowledge is still valuable as the correct Python environment needs to be activated (as in the other cases as well), but the app itself guides the user through the process.


## Worflow to Generate and Benchmark Models
If you wish to reproduce the results presented here, or if you wish to expand the variety of classifiers for further benchmarking, here's a suggested workflow:

1. [`create_csvs.ipynb`](files/create_csvs.ipynb): collate all the separate CSV files containing the necessary data into one large dataset file, alongside generating figures summarising dataset properties.
2. [`hyperparameters_optimisation.ipynb`](files/hyperparameters_optimisation.ipynb): optimise the hyperparameters of the supervised classifiers (*k*NN, GNB, polynomial and RBF SVM) on the newly created dataset.
3. [`benchmark.ipynb`](files/benchmark.ipynb): benchmark the models against established definitions.
    - [`compare_ru.ipynb`](files/compare_ru.ipynb): for further benchmarking with the [Rivas-Ubach *et al.* (2018)](https://doi.org/10.1021/acs.analchem.8b00529) validation dataset.
5. [`model_generation.ipynb`](files/model_generation.ipynb): generate and serialise ("pickle") the SSEFCCA models. This will also generate interactive 3D decision boundaries plots as HTML files using [Plotly](https://plotly.com/python/).
6. [`3d_plots.ipynb`](files/3d_plots.ipynb): generate more 3D plots of the decision boundaries for presentations.

All modules used can be found in the [`files/requirements.txt`](files/requirements.txt) file.

### Parallel computing
Given the sheer size of the final dataset, it was necessary to swap to cluster computing. This was done using the scripts in the `parallel_*` folders on [Eddie](https://information-services.ed.ac.uk/research-support/research-computing/ecdf/high-performance-computing), the University of Edinburgh's Research Compute Cluster, running on Rocky Linux 9 using Sun Grid Engine (SGE) as its job scheduler.
