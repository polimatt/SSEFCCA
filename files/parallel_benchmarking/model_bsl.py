import argparse
# Data manipulation and numerical computations
import numpy as np
import pandas as pd

# Machine learning classifiers and metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import brier_score_loss
# Custom functions for SSEFCCA (Supervised Semi-Empirical Formula Compound Class Assignment)
import sys
sys.path.append('..')
import ssefcca_functions as sfc

crs = __import__('create_rdm_sets')

from warnings import filterwarnings
# Suppress expected RuntimeWarnings from np.nanmean when calculating class-wise accuracies with all NaN values
filterwarnings('ignore', module='numpy', category=RuntimeWarning)

#%%
# argparse
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--index', help='Choose the index')
parser.add_argument('-m', '--model', help='Choose the model from: knn, gnb, poly_svm, rbf_svm')
parser.add_argument('-c', '--classes', help='Choose the class definitions from: with_pahs')
args = parser.parse_args()
# print(args.index,args.model,args.classes)


#%%

data_folder = '../data'
benchmarking_folder = f'{data_folder}/csv_files/benchmarking'
traintest_folder = f'{benchmarking_folder}/train-test_datasets'
bsl_output_folder = f'{benchmarking_folder}/bsl'

#%%

clf_models = {
    'knn':sfc.clf_pipeline(KNeighborsClassifier(**(sfc.knn_param_3d if args.classes!='with_pahs' else sfc.knn_param_3d_with_PAHs))),
    'gnb':sfc.clf_pipeline(GaussianNB()),
    'poly_svm':sfc.clf_pipeline(SVC(**(sfc.poly_svm_param_3d if args.classes!='with_pahs' else sfc.poly_svm_param_3d_with_PAHs))),
    'rbf_svm':sfc.clf_pipeline(SVC(**(sfc.rbf_svm_param_3d if args.classes!='with_pahs' else sfc.rbf_svm_param_3d_with_PAHs))),

    'knn_multi':sfc.clf_pipeline(KNeighborsClassifier(**(sfc.knn_param_multi))),
    'gnb_multi':sfc.clf_pipeline(GaussianNB()),
    'poly_svm_multi':sfc.clf_pipeline(SVC(**(sfc.poly_svm_param_multi))),
    'rbf_svm_multi':sfc.clf_pipeline(SVC(**(sfc.rbf_svm_param_multi))),
}

match args.model:
    case 'knn_multi' | 'gnb_multi' | 'poly_svm_multi' | 'rbf_svm_multi' :
        columns = ['O/C', 'H/C', 'N/C', 'P/C']
    case _:
        columns = ['O/C', 'H/C', 'N/C']

test_df = pd.read_csv(f'{traintest_folder}/{args.index}--test_{args.classes}.csv')
X_test = test_df[columns].to_numpy()
y_test = test_df['y'].to_numpy()
weights_test = test_df['weights'].to_numpy()

#%%
# get y_pred
train_df = pd.read_csv(f'{traintest_folder}/{args.index}--train_{args.classes}.csv')
X_train = train_df[columns].to_numpy()
y_train = train_df['y'].to_numpy()
weights_train = train_df['weights'].to_numpy()

fit_args = dict(X=X_train, y=y_train)
if 'knn' not in args.model:
    fit_args['clf__sample_weight'] = weights_train

clf = clf_models[args.model].fit(**fit_args)

y_proba = clf.predict_proba(X_test)


#%%
# get bsl
bsl = str(brier_score_loss(y_test, y_proba, sample_weight=weights_test))

model_string = args.model

with open(f'{bsl_output_folder}/{args.classes}_{model_string}_bsl--{args.index}.txt', 'w') as f:
  f.write(bsl)