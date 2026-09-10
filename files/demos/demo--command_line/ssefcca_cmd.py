import pandas as pd
import numpy as np
import os
import argparse
from pickle import load
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

parser = argparse.ArgumentParser(prog = 'ssefcca_cmd.py',
                                 description='script to assign the compound class of empirical formulae contained in CSV files.')

parser.add_argument('dirpath')
parser.add_argument('-mp','--modelpath')
parser.add_argument('-op','--outputpath')

args = parser.parse_args()

# set defaults if arguments not specified
model_path = '../../data/models/svm_poly.pkl' if not args.modelpath else args.modelpath
outputpath = f'{args.dirpath}--assigned' if not args.outputpath else args.outputpath

if not os.path.isdir(outputpath):
    os.mkdir(outputpath)

with open(model_path, 'rb') as f:
    model = load(f)

def compound_classification(data_df:pd.DataFrame,model:CalibratedClassifierCV|Pipeline,
                            molec_form_column = 'Molecular Formula',
                            ratios_needed:list = ['O/C','H/C','N/C'],
                            isotope_pairs:list[list] = [['C','13C'],['H','2H'],['N','15N'],['O','18O'],['S','34S']]) -> pd.DataFrame:

    assert isinstance(model,CalibratedClassifierCV|Pipeline), '`model` must be a `skl.calibration.CalibratedClassifierCV` or `skl.pipeline.Pipeline`'
    assert molec_form_column in data_df.columns, '`molec_form_column` must be the column of the `data_df` dataframe containing the assigned empirical formulae.'

    # copy the dataframe into another variable the function will be working on
    # substitute all NaNs with zeros and only select the rows with assigned empirical formulae
    assigned_df = data_df.copy()
    assigned_df = assigned_df[~assigned_df[molec_form_column].isna()]
    assigned_df.fillna(0,inplace=True)

    # check that all necessary stoichiometric ratios are present in the df; add them in if not
    for ratio in ratios_needed:
        if ratio not in assigned_df.columns:
            elements = ratio.split('/')

            numerator = assigned_df[elements[0]]
            denominator = assigned_df[elements[1]]

            # account for isotopes
            for pair in isotope_pairs:
                if elements[0] == pair[0] and pair[1] in assigned_df.columns:
                    numerator += assigned_df[pair[1]]

                if elements[1] == pair[0] and pair[1] in assigned_df.columns:
                    denominator += assigned_df[pair[1]]
            
            assigned_df[ratio] = numerator / denominator
    
    # assign compound classes, their probabilities, and add them as new columns
    assignments = model.predict(assigned_df[ratios_needed].values)
    max_probabilities = np.max(model.predict_proba(assigned_df[ratios_needed].values),axis=1)
    assigned_df['compound_class'] = assignments
    assigned_df['probability'] = max_probabilities

    return assigned_df

files_list = [x for x in os.listdir(args.dirpath) if x.endswith('.csv')]

for file in files_list:
    file_path = f'{args.dirpath}/{file}'
    data_df = pd.read_csv(file_path)
    data_df = compound_classification(data_df,model)
    data_df.to_csv(f'{outputpath}/{file.replace('.csv','--assigned.csv')}')