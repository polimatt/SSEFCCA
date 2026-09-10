# %%
import argparse
from pickle import dump

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import sys
sys.path.append('..')
import ssefcca_functions as sfc
#%%

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', help='Choose the model from: knn, gnb, poly_svm, rbf_svm')
args = parser.parse_args()


# %%
data_dir = '../data'

df_with_pahs_path = f'{data_dir}/csv_files/new_vk_areas_with_pahs.csv'

decision_boundaries_output_folder = f'{data_dir}/plots/decision_boundaries'
models_output_folder = f'{data_dir}/models'

weighted = True
show_title = False

# %%
def pickle_model(clf, filepath):
    with open(filepath, 'wb') as f:
        dump(clf, f, protocol=5)


def create_3D_mesh(data, model, dims, number_of_points=50) -> pd.DataFrame:
    # https://www.kaggle.com/code/frankmollard/3d-decision-boundaries
    '''
    Create a 3D meshgrid for decision boundary visualization.
    '''
    data=data.copy()
    meshes_2d={}
    for i in range(len(dims)):
        meshes_2d.update({f'{dims[i]}': np.linspace(np.min(data[:,i]), np.max(data[:,i]), number_of_points)})

    # create meshgrid
    meshes_3d = np.meshgrid(*[x for x in meshes_2d.values()],
                            indexing='ij')

    stacked_mesh = np.vstack([x.ravel() for x in meshes_3d]).T  

    mesh_df = pd.DataFrame(stacked_mesh, columns=dims)

    preds = model.predict_proba(stacked_mesh)
    argmax = np.argmax(preds, axis=1)

    highest_probs = np.max(preds, axis=1)

    mesh_df['category'] = model.classes_[argmax]
    mesh_df['probability'] = highest_probs

    return mesh_df.sort_values(by=['category']).reset_index(drop=True)


def create_fig_for_3d_decision_boundary(mesh_df:pd.DataFrame, title:str=None, dims:list=None, alpha=.8, squareSize=10, colours=None) -> go.Figure:
    vizList = []
    for compound_class in mesh_df['category'].unique():

        i = mesh_df['category'] == compound_class

        compound_class_formatted = compound_class.capitalize().replace('Pah', 'PAH').replace('_',' ')

        globals()[f'D_{compound_class}']=go.Scatter3d(

            x = mesh_df.loc[i, dims[0]],
            y = mesh_df.loc[i, dims[1]],
            z = mesh_df.loc[i, dims[2]],

            name = f'{compound_class_formatted}-like',

            hovertemplate = dims[0] + ': %{x} <br>' + \
                            dims[1] + ': %{y} <br>' + \
                            dims[2] + ': %{z} <br>',
                            # 'category: {0} <br>'.format(compound_class_formatted),

            mode='markers',
            marker=dict(
                size=squareSize,
                color=colours[compound_class],           
                opacity=alpha,#*mesh_df.loc[i, 'probability'].to_numpy(),
                symbol='square'
            )
        )

        vizList.append(globals()[f'D_{compound_class}']) 

    fig = go.Figure(data=vizList,)

    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=-1, y=-1.75, z=1.5)
    )

    fig.update_layout(
        scene_camera=camera,
        autosize=True,
        template='plotly_white',
        scene={
            'xaxis_title':dims[0],
            'yaxis_title':dims[1],
            'zaxis_title':dims[2]
        },
        width=1800, height=900,
        showlegend=True,
        title = title
    )

    return fig


def DecisionBoundaryDisplay_3D(data, model, dims, alpha=.8, title=None, squareSize=10, number_of_points=50,
                               colours = sfc.vk_region_colours,
                               savepath=None,show=True):
    # https://www.kaggle.com/code/frankmollard/3d-decision-boundaries
    '''
    Visualize 3D decision boundaries using Plotly.
    '''

    data = data.copy()

    mesh_df = create_3D_mesh(data=data, model=model, dims=dims, number_of_points=number_of_points)

    fig = create_fig_for_3d_decision_boundary(mesh_df=mesh_df, title=title, dims=dims, alpha=alpha, squareSize=squareSize, colours=colours)

    if show: fig.show()
    if savepath: fig.write_html(savepath)

# %%
df_with_pahs = pd.read_csv(df_with_pahs_path)


# %%
columns_3d = ['O/C','H/C','N/C']

X_with_pahs_3d = df_with_pahs[columns_3d].to_numpy()
y_with_pahs = df_with_pahs['category'].to_numpy()
weights_with_pahs = df_with_pahs['weights'].to_numpy() if weighted else None

# %%

match args.model:
    case 'knn':
        clf = sfc.clf_pipeline(KNeighborsClassifier(**sfc.knn_param_3d_with_PAHs)).fit(X_with_pahs_3d, y_with_pahs)
        title = 'K-Nearest Neighbours'
    case 'gnb':
        clf = sfc.clf_pipeline(GaussianNB()).fit(X_with_pahs_3d, y_with_pahs, clf__sample_weight=weights_with_pahs if weighted else None)
        title = 'Gaussian Naive Bayes'
    case 'poly_svm':
        clf = sfc.clf_pipeline(SVC(**sfc.svm_poly_param_3d_with_PAHs,probability=True)).fit(X_with_pahs_3d, y_with_pahs, clf__sample_weight=weights_with_pahs if weighted else None)
        title = 'Polynomial SVM'
    case 'rbf_svm':
        clf = sfc.clf_pipeline(SVC(**sfc.svm_rbf_param_3d_with_PAHs,probability=True)).fit(X_with_pahs_3d, y_with_pahs, clf__sample_weight=weights_with_pahs if weighted else None)
        title = 'RBF SVM'

# %%

pickle_model(clf,f'{models_output_folder}/{args.model}.pkl')

DecisionBoundaryDisplay_3D(X_with_pahs_3d,
                           clf,
                           dims=columns_3d,
                           title=f'{title} 3D Decision Boundary\non van Krevelen Diagram',
                           savepath=f'{decision_boundaries_output_folder}/{args.model}_3d_vk_decision_boundary_with_pahs.html',
                           show=False,)
