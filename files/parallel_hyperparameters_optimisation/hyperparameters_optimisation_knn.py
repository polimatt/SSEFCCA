# %%
import pandas as pd
from pickle import load, dump
from os.path import exists

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# %%
data_folder = '../data'
df_path = f'{data_folder}/csv_files/new_vk_areas_df.csv'
df_with_pahs_path = f'{data_folder}/csv_files/new_vk_areas_with_pahs.csv'

weighted = True

pkl_file_path = f'{data_folder}/hyperparameters/hyperparameters.pkl'

# %%
df = pd.read_csv(df_path)
df_with_pahs = pd.read_csv(df_with_pahs_path)

# %%
columns_3d = ['O/C','H/C','N/C']
X_3d = df[columns_3d].to_numpy()
X_3d_scaled = StandardScaler().fit_transform(X_3d)

y = df['category'].to_numpy()
weights = df['weights'].to_numpy() if weighted else None

X_3d_with_pahs = df_with_pahs[columns_3d].to_numpy()
X_3d_with_pahs_scaled = StandardScaler().fit_transform(X_3d_with_pahs)

y_with_pahs = df_with_pahs['category'].to_numpy()

weights_with_pahs = df_with_pahs['weights'].to_numpy() if weighted else None

# %%
def find_best_hyperparams(clf,params_dictionary,X,y,sample_weight=None,refit=True, verbose=1, scoring='balanced_accuracy'):
    grid = GridSearchCV(clf,params_dictionary, refit=refit, verbose=verbose, scoring=scoring, n_jobs=-1)
    grid.fit(**dict(X=X,y=y,sample_weight=sample_weight) if sample_weight is not None else dict(X=X,y=y))
    return grid

# %% [markdown]
# ## Optimise KNN

# %%
param_grid_knn = {
    'weights':['distance'],
    'n_neighbors': [1,5,10,50,100,200],
    }

# %%
grid_knn_3d = find_best_hyperparams(KNeighborsClassifier(),param_grid_knn,X_3d_scaled,y)
print('3D KNN best_estimator_:', grid_knn_3d.best_estimator_)

# %%
grid_knn_pahs_3d = find_best_hyperparams(KNeighborsClassifier(),param_grid_knn,X_3d_with_pahs_scaled,y_with_pahs)
print('3D KNN with PAHs best_estimator_:', grid_knn_pahs_3d.best_estimator_)


# %%
text_file = open(f'{data_folder}/hyperparameters/hyperparameters_knn.txt', 'w')

text_file.write(f'''3D KNN best_estimator_: {grid_knn_3d.best_estimator_}
3D KNN with PAHs best_estimator_: {grid_knn_pahs_3d.best_estimator_}

''')

text_file.close()

# %%
hyperparameters = {
    'knn_3d':grid_knn_3d.best_params_,
    'knn_pahs_3d':grid_knn_pahs_3d.best_params_,
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
