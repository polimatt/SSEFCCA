This demo aims at demonstrating how the script compound_class_prediction.py can be used from the command line to predict the compound class of molecular formulae contained in various CSV files derived from mass spectra analysis.



First of all, it is important to activate a suitable Python environment. In this case, a list of modules used is available in the `../requirements.txt` file.
To create a conda environment from it, the following command may be used:
```console
$ conda create --name mlacca --file <path to requirements.txt>
```

The command used to activate the mlacca_env conda environment is:

```console
$ conda activate mlacca
```

The full syntax to operate the script from the command line is:

```console
python <path to the script>/mlacca_cmd.py [path to the directory containing the CSV files] -mp [path to the prediction model] -ndp [path of the directory where the CSV files with the predicted compound classes will be saved]
```


The named arguments are optional: the path to the model is hardcoded into the script, but can be changed with any IDE or also with a text editor such as Notepad; the path of the new directory becomes the path of the old folder + "--assigned" unless otherwise specified.

If the current working directory is the one containing the script, and if within the same directory the folder with the CSV files is also present and called `folder`, then the command will look like:

```console
$ python mlacca_cmd.py folder
```

This will then save the outputs into a folder--assigned directory.

To sum up, the commands used in this demo following the creation of a `mlacca` Python environment are:

```console
conda activate mlacca
python mlacca_cmd.py folder
```

