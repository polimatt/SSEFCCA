# %%
import pandas as pd
from pickle import load, dump
from os.path import exists

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

# %%
data_folder = '../data'
df_with_pahs_path = f'{data_folder}/csv_files/new_vk_areas_with_pahs.csv'

weighted = True

pkl_file_path = f'{data_folder}/hyperparameters/hyperparameters.pkl'

# %%
df_with_pahs = pd.read_csv(df_with_pahs_path)

# %%
columns_multi = ['O/C', 'H/C', 'N/C', 'P/C']
X_multi_with_pahs = df_with_pahs[columns_multi].to_numpy()
X_multi_with_pahs_scaled = StandardScaler().fit_transform(X_multi_with_pahs)

y_with_pahs = df_with_pahs['category'].to_numpy()
weights = df_with_pahs['weights'].to_numpy() if weighted else None

# %%
def find_best_hyperparams(clf,params_dictionary,X,y,sample_weight=None,refit=True, verbose=1, scoring='balanced_accuracy'):
    grid = GridSearchCV(clf,params_dictionary, refit=refit, verbose=verbose, scoring=scoring, n_jobs=-1)
    grid.fit(**dict(X=X,y=y,sample_weight=sample_weight) if sample_weight is not None else dict(X=X,y=y))
    return grid

# %%
# choose C and gamma hyperparameters for SVM with polynomial kernel for the multi case
# https://www.geeksforgeeks.org/machine-learning/svm-hyperparameter-tuning-using-gridsearchcv-ml/

param_grid_poly_svm = {
    'kernel':['poly'],
    'degree': list(range(1,4)),
    'C': [10**x for x in range(4)],
    'gamma': ['auto'] + [10**(-x) for x in range(4)],
    }
param_grid_poly_svm

# %%
grid_poly_svm_multi = find_best_hyperparams(SVC(),param_grid_poly_svm,X_multi_with_pahs_scaled,y_with_pahs,sample_weight=weights)
print('multi poly SVM best_estimator_:', grid_poly_svm_multi.best_estimator_)

# %%
text_file = open(f'{data_folder}/hyperparameters/hyperparameters_poly_svm_nopah.txt', 'w')

text_file.write(f'''Multi SVM (poly) best_estimator_: {grid_poly_svm_multi.best_estimator_}''')

text_file.close()

# %%
hyperparameters = {
    'poly_svm_multi':grid_poly_svm_multi.best_params_,
}

if exists(pkl_file_path):
    with open(pkl_file_path, 'rb') as f:
        pkl_file = load(f)
else: pkl_file = {}

print(pkl_file)

for key in hyperparameters:
    pkl_file[key] = hyperparameters[key]

with open(pkl_file_path, 'wb') as f:
    dump(pkl_file, f)
