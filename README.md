# MLACCA: Machine Learning-Assisted Compound Class Assignment

Here, code is presented to create and use supervised classification models for elemental ratios (O/C, H/C, and N/C)-based compound class assignment derived from mass spectrometry-derived molecular formula assignment.

All material is in the `files` folder. Of special interest may be the pickled models contained in `files/data/models` and the demos presented in `files/demos`. The demos are organised so that end users may benefit from this method (almost) regardless of their coding skill level:
- In the `files/demo/demo--jupyter` folder, the demo is aimed at those who want to integrate the classification models directly into their own Python-based MS data analysis pipeline.
- In the `files/demo/demo--command_line` folder, the demo explains how to use this method from the command line, thus requiring some familiarity with the command line interface but relatively little Python knowledge.
- Finally, in the `files/demo/demo--app` a Python-based GUI application is presented. Here, some IT knowledge is still valuable as the correct Python environment needs to be activated (as in the other cases as well), but the app itself guides the user through the process.

All modules used can be found in the `files/requirements.txt` file.
