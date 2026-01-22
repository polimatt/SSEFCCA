# MLACCA: Machine Learning Assisted Compound Class Assignment

Here, code is presented to create and apply supervised classification models for compound class assignment based on elemental ratios (O/C, H/C, and N/C) derived from empirical formula assignment of mass spectrometry data.

All material is present in the ```files``` folder. Of special interest may be the pickled models contained in ```files/vk_definitions/models``` and the demos presented in ```files/demos```. The demos are organised so that end users may benefit from this method (almost) regardless of their coding skill level:
- In the ```files/demo--jupyter``` folder, the demo is aimed at those who want to integrate the classification models directly into their own Python-based MS data analysis pipeline.
- In the ```files/demo--command_line``` folder, the demo explains how to use this method from the Command Prompt, thus requiring some familiarity with the command line interface but relatively little Python knowledge.
- Finally, in the ```files/demo--app``` a Python-based GUI application is presented. Here, some IT knowledge is still valuable as the correct Python environment needs to be activated (as in the other cases as well), but the app itself guides the user through the process.

All modules used can be found in the conda backup file ```files/mlacca_env.yml```.
