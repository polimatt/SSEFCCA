![mlacca logo]([https://github.com/polimatt/MLACCA/code/demos/demo--app/files_for_app/logo/logo/mlacca.png])  

# MLACCA: Machine Learning Assisted Compound Class Assignment

Here, code is presented to create and apply supervised classification models for compound class assignment based on elemental ratios (O/C, H/C, and N/C) derived from empirical formula assignment of mass spectrometry data.

All material is present in the folder ```code```. Of special interest may be the pickled models contained in ```code/vk_definitions/models``` and the demos presented in ```code/demos```. The demos are organised so that end users may benefit from this method (almost) regardless of their coding skill level:
- In the ```code/demo--jupyter``` folder, the demo is aimed at those who want to integrate the classification models directly into their own Python-based MS data analysis pipeline.
- In the ```code/demo--command_line``` folder, the demo explains how to use this method from the Command Prompt, thus requiring some familiarity with the command line interface but relatively little Python knowledge.
- Finally, in the ```code/demo--app``` a Python-based GUI application is presented. Here, some IT knowledge is still valuable as the correct Python environment needs to be activated (as in the other cases as well), but the app itself guides the user through the process.

All modules used can be found in the conda backup file ```code/mlacca_env.yaml```.
