This demo aims at demonstrating how the script compound\_class\_prediction.py can be used from the command line to predict the compound class of molecular formulae contained in various CSV files derived from mass spectra analysis.



First of all, it is important to activate a suitable Python environment. In this case, a list of modules used is available in the ..\\mlacca\_env.yaml file. The command used to activate the mlacca\_env conda environment is:



```conda activate vk_definitions```



The full syntax to operate the script from the command line is:



```python \[path to the script]\\compound\_class\_prediction.py \[path to the directory containing the CSV files] -mp \[path to the prediction model] -ndp \[path of the directory where the CSV files with the predicted compound classes will be saved]```



Only the python ```\[path to the script]\\compound\_class\_prediction.py \[path to the directory containing the CSV files]``` part is mandatory, the rest is optional: the path to the model is hardcoded into the script, but can be changed with any IDE or also with a text editor such as Notepad; the path of the new directory becomes the path of the old folder + "--predicted" unless specified.



In the case in which the Command Prompt is already pointing at the directory containing the script, and in the case in which within the same directory the folder with the CSV files is also present, the command will look like:



```python compound_class_prediction.py folder```



This will then save the outputs into a folder--predicted directory.



To sum up, the commands used in this demo are:



```

conda activate vk_definitions
python compound_class_prediction.py folder

```

