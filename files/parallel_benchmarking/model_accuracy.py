import argparse
# Data manipulation and numerical computations
import numpy as np
import pandas as pd

# Machine learning classifiers and metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
# Custom functions for SSEFCCA (Supervised Semi-Empirical Formula Compound Class Assignment)
import sys
sys.path.append('..')
import ssefcca_functions as sfc

crs = __import__('create_rdm_sets')

from warnings import filterwarnings
# Suppress expected sklearn UserWarnings when using different class definitions
filterwarnings('ignore', module='sklearn', category=UserWarning)
# Suppress expected RuntimeWarnings from np.nanmean when calculating class-wise accuracies with all NaN values
filterwarnings('ignore', module='numpy', category=RuntimeWarning)

#%%
# argparse
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--index', help='Choose the index')
parser.add_argument('-m', '--model', help='Choose the model from: zero_rule, dbccr, 3d_hybrid, hybrid_complete, 3d_mscc, mscc, knn, gnb, poly_svm, rbf_svm')
parser.add_argument('-c', '--classes', help='Choose the class definitions from: non_r-u, r-u, with_pahs')
parser.add_argument('-w', '--weighted', help='Choose whether or not to weight the training for the supervised classifier (options: True, False)')
args = parser.parse_args()
# print(args.index,args.model,args.classes)
if args.weighted is not None:
    match args.weighted:
        case 'True': args.weighted = True
        case 'False': args.weighted = False

#%%

data_folder = '../data'
benchmarking_folder = f'{data_folder}/csv_files/benchmarking'
traintest_folder = f'{benchmarking_folder}/train-test_datasets'
acc_output_folder = f'{benchmarking_folder}/accuracies'

#%%
lit_models = {
    'zero_rule':[],
    'dbccr':sfc.laszakovits_mackay_areas,
    '3d_hybrid':sfc.hybrid_areas,
    'hybrid_complete':sfc.hybrid_areas_complete,
    '3d_mscc':sfc.rivasubach_areas,
    'mscc':sfc.rivasubach_areas_complete
}
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

['knn','gnb','poly_svm','rbf_svm',]

match args.model:
    case 'dbccr':
        columns = ['O/C','H/C']

    case  'hybrid_complete' | 'mscc':
        columns = ['O/C', 'H/C', 'N/C', 'P/C', 'N/P']

    case 'knn_multi' | 'gnb_multi' | 'poly_svm_multi' | 'rbf_svm_multi':
        columns = ['O/C', 'H/C', 'N/C', 'P/C']

    case _:
        columns = ['O/C', 'H/C', 'N/C']

test_df = pd.read_csv(f'{traintest_folder}/{args.index}--test_{args.classes}.csv')
X_test = test_df[columns].to_numpy()
y_test = test_df['y'].to_numpy()
weights_test = test_df['weights'].to_numpy()

#%%
# get y_pred
if args.model in clf_models:
    train_df = pd.read_csv(f'{traintest_folder}/{args.index}--train_{args.classes}.csv')
    X_train = train_df[columns].to_numpy()
    y_train = train_df['y'].to_numpy()
    weights_train = train_df['weights'].to_numpy()

    fit_args = dict(X=X_train, y=y_train)
    if args.weighted:
        fit_args['clf__sample_weight'] = weights_train

    clf = clf_models[args.model].fit(**fit_args)

    y_pred = clf.predict(X_test)


elif args.model in lit_models:
    if args.model == 'zero_rule':
        zero_rule_class = np.unique(y_test)[np.argmax([len(np.where(y_test == cat)[0]) for cat in np.unique(y_test)])]
        y_pred = [zero_rule_class] * len(y_test)

    else:
        y_pred = sfc.molecclass(pd.DataFrame(X_test, columns=columns),
                                areas=lit_models[args.model],
                                dims=columns)


#%%
# get accuracy
acc = {'raw':None,'balanced':None}
for accuracy_type in [[accuracy_score, dict(normalize=True),'raw'],
                      [balanced_accuracy_score, dict(sample_weight=weights_test),'balanced']]:
  
    acc[accuracy_type[2]] = [accuracy_type[0](y_true=y_test,
                                              y_pred=y_pred,
                                              **accuracy_type[1])]

model_string = args.model
if args.model in clf_models and 'knn' not in args.model and 'multi' not in args.model:
    if args.weighted:
        model_string += '_w'
    else:
        model_string += '_u'

acc = pd.DataFrame(acc)
acc.to_csv(f'{acc_output_folder}/{args.classes}_{model_string}_accuracies--{args.index}.csv',index=False)

class_acc = sfc.class_wise_accuracy(y_true=y_test, y_pred=y_pred)
class_acc = {x:[class_acc[x]] for x in class_acc}
class_acc = pd.DataFrame(class_acc)
class_acc.to_csv(f'{acc_output_folder}/{args.classes}_{model_string}_classwiseaccuracies--{args.index}.csv',index=False)

conf_mat_classes = np.unique([y_pred,y_test])
conf_mat = confusion_matrix(y_true=y_test,y_pred=y_pred,normalize='true')
conf_mat = pd.DataFrame(conf_mat,columns=conf_mat_classes,index=conf_mat_classes)
conf_mat.to_csv(f'{acc_output_folder}/{args.classes}_{model_string}_confusionmatrix--{args.index}.csv')
