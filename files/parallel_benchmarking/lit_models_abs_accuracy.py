
# Data manipulation and numerical computations
import numpy as np
import pandas as pd

# Plotting and visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

# Custom functions for SSEFCCA (Supervised Semi-Empirical Formula Compound Class Assignment)
import sys
sys.path.append('..')
import ssefcca_functions as sfc

from warnings import filterwarnings
# Suppress expected sklearn UserWarnings when using different class definitions
filterwarnings('ignore', module='sklearn', category=UserWarning)
# Suppress expected RuntimeWarnings from np.nanmean when calculating class-wise accuracies with all NaN values
filterwarnings('ignore', module='numpy', category=RuntimeWarning)

#%%
# Configuration settings and file paths

# Output folder for comparison plots and CSV files
data_folder = '../data'
comparison_plots_output_folder = f'{data_folder}/plots/comparisons'
csv_folder = f'{data_folder}/csv_files'

# Path to the dataset containing van Krevelen area assignments
df_path = f'{csv_folder}/new_vk_areas_df.csv'

# Number of random train-test splits
repeats = 20

# Model configuration flags
weighted = True  # Whether to use class weights to handle imbalanced datasets
show_title = False  # Whether to display titles on plots (False for publication-ready figures)
stratify = True # Whether to stratify the train-test split
override_save = True # True: save all figures and outputs, False don't save anything

df = pd.read_csv(df_path)

# Custom utility functions for model evaluation and visualization

def test_model_accuracy(model, X, y, repeats=50, classifier_name='[DEFAULT]', title=None,
                        xlabel=None, ylabel=None, colors_dict=None):
    '''
    Test a machine learning model's accuracy over multiple random train-test splits.
    
    Parameters:
    -----------
    model : sklearn classifier
        The model to test
    X : array-like
        Feature matrix
    y : array-like
        Target labels
    repeats : int
        Number of random train-test splits to perform
    classifier_name : str
        Name of the classifier for reporting
    title, xlabel, ylabel : str, optional
        Plot labels (only used for 2D feature spaces)
    colors_dict : dict, optional
        Color mapping for categories in decision boundary plot
    
    Returns:
    --------
    Prints accuracy statistics and optionally creates decision boundary plot (for 2D data)
    '''
    accuracies = []

    # Repeat training and testing with random splits
    for i in range(repeats):
        X_rdm_train, X_rdm_test, y_rdm_train, y_rdm_test, \
        weights_rdm_train, weights_rdm_test = sfc.train_test(X, y, random_state=None, stratify_yn=stratify)

        # Fit model with sample weights if weighted mode is enabled
        fitted_model = model.fit(X_rdm_train, y_rdm_train, sample_weight=weights_rdm_train if weighted else None)

        # Calculate balanced accuracy score
        accuracies.append(balanced_accuracy_score(y_true=y_rdm_test,
                                                  y_pred=fitted_model.predict(X_rdm_test),
                                                  sample_weight=weights_rdm_test if weighted else None))

        # Keep track of the best performing model for visualization
        if accuracies[i] == np.max(accuracies):
            max_model = fitted_model
            X_rdm_train_best = X_rdm_train
            y_rdm_train_best = y_rdm_train

    # Print summary statistics
    print(f'''Average accuracy of the {classifier_name} classifier trained on {len(X_rdm_train)} samples (run {repeats} times): {np.mean(accuracies)} +/- {np.std(accuracies)}
Max accuracy: {np.max(accuracies)}
Min accuracy: {np.min(accuracies)}
Random Accuracy: {1/len(np.unique(y_rdm_train))}''')

    # Create decision boundary plot for 2D feature spaces
    if np.shape(X)[1] == 2:
        sfc.draw_decision_boundary_plot(max_model,
                                    X_rdm_train_best,
                                    categories=np.unique(y_rdm_train_best),
                                    xlabel=xlabel,
                                    ylabel=ylabel,
                                    colors_dict=colors_dict,
                                    title=title)


def get_rounding(x:float):
    if x < 1 and x != 0:
        string = str(x)
        string = string.split('.')

        for i in range(len(string[1])):
            if i == len(string[1])-1:
                return i + 1

            elif string[1][i] != '0':
                if int(string[1][i+1]) >= 5 and string[1][i]=='9':
                    return i
                else: return i + 1
    else:
        return -len(str(int(np.round(x,0))))+1

def get_right_dec_figs_in_string(n,dec_places):
    dec_figs = len(str(n).split('.')[1])
    if dec_figs < dec_places:
        return str(n) + ''.join(['0']*(dec_places-dec_figs))
    else:
        return str(n)

def plus_minus(mean,std):
    std_dec_places = [get_rounding(x) if x!=0 else 0 for x in std]

    list_of_strings = []

    for x, y, z in zip(mean,std,std_dec_places):
        if y < 1 and y != 0:
            list_of_strings.append(f'{get_right_dec_figs_in_string(np.round(x,z),z)} ± {get_right_dec_figs_in_string(np.round(y,z),z)}')

        elif y == 0:
            list_of_strings.append(f'{get_right_dec_figs_in_string(np.round(x,z),z)}')

        else:
            list_of_strings.append(f'{int(np.round(x,z))} ± {int(np.round(y,z))}')

    for i in range(len(list_of_strings)):
        list_of_strings[i] = list_of_strings[i].replace('.0 ',' ')
        if list_of_strings[i].endswith('.0'): list_of_strings[i] = list_of_strings[i][:-2]

    
    return list_of_strings


def draw_confusion_matrix(matrix,categories,
                          cmap='afmhot_r',text=True,text_matrix=None,
                          fontsize=10):
    from matplotlib.colors import Normalize
    fig, ax = plt.subplots()

    im = ax.imshow(matrix,
                   norm=Normalize(0, 100),cmap=cmap)

    if text:
        for (i,j), _ in np.ndenumerate(matrix):
            ax.text(j,i,
                    get_right_dec_figs_in_string(100*matrix[j,i],2) if text_matrix is None else text_matrix[i,j],
                    ha='center',va='center',
                    c='#fff',
                    fontsize=fontsize,
                    path_effects=[pe.withStroke(linewidth=2,foreground='#000')])


    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.set_label('Percentage [%]', fontsize=12)

    ax.set_xlabel('Predicted label',fontsize=12)
    ax.set_ylabel('True label',fontsize=12)

    ax.set_xticks(range(len(categories)),labels=[x.replace('_',' ') for x in categories],ha='right',rotation=30)
    ax.set_yticks(range(len(categories)),labels=[x.replace('_',' ') for x in categories])
    return fig, ax


def set_y_ticks(data_dict, ax, major_ticks_interval=0.1, minor_ticks_interval=0.05, rounding=2):
    '''
    Set y-axis tick marks with automatic limits based on data range.
    
    Parameters:
    -----------
    data_dict : dict
        Data dictionary to determine axis limits
    ax : matplotlib axis
        Axis object to modify
    major_ticks_interval : float
        Spacing for major tick marks
    minor_ticks_interval : float
        Spacing for minor tick marks
    rounding : int
        Decimal places for rounding the lower limit
    '''
    # Calculate y-axis limits based on data minimum
    rounded_min = np.round(np.min(list(data_dict.values())))
    rounded_min = rounded_min - (rounded_min % major_ticks_interval)
    # rounded_max = np.round(np.min(list(data_dict.values())))
    # rounded_max = rounded_min - (rounded_min % major_ticks_interval) + major_ticks_interval

    ylim = (rounded_min, 100)
    ax.set_ylim(ylim)
    
    # Set major and minor tick marks
    ax.set_yticks(np.arange(np.round(np.floor(1e2 * ylim[0]) / 1e2, 1), ylim[1] + 1e-2, major_ticks_interval))
    ax.tick_params(axis='y', which='minor')
    ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(minor_ticks_interval))

#%%

# Define feature sets for different dimensionalities of van Krevelen space

# 2D van Krevelen diagram (commonly used in literature)
columns_2d = ['O/C', 'H/C']

# 3D extended van Krevelen space (includes N/C)
columns_3d = ['O/C', 'H/C', 'N/C']
X_3d = df[columns_3d].copy().to_numpy()

# Multi-dimensional feature space (includes P/C and N/P ratios)
columns_multi = ['O/C', 'H/C', 'N/C', 'P/C', 'N/P']

# True compound class assignments
y = df['category'].copy().to_numpy()

# Sample weights for handling class imbalance (if weighted mode is enabled)
weights = df['weights'].copy().to_numpy() if weighted else None

# Create Rivas-Ubach categories (merge tannin and lignin into phytochemical)
y_ru = df['category'].copy().to_numpy()
y_ru[[x in ['tannin', 'lignin'] for x in y_ru]] = 'phytochemical'

# Compute class weights for Rivas-Ubach categories
class_weights_ru = compute_class_weight('balanced', classes=np.unique(y_ru), y=y_ru)
weights_dict_ru = {cls: weight for cls, weight in zip(np.unique(y_ru), class_weights_ru)}
weights_ru = [weights_dict_ru[cat] for cat in y_ru]

# Color scheme for each category in multi-class plots
multiclass_colors = [sfc.vk_region_colours[cat] for cat in np.unique(y)]


#%%

# Compare accuracy of literature-based classification models

lit_acc_comparison = {'lit_model': [], 'accuracy': [], 'balanced_accuracy': []}

confusion_matrices = []

# Test each literature model and baseline
combos = [['Zero rule', y],  # Baseline: predict most frequent class
          ['DBCCR', sfc.laszakovits_mackay_areas, columns_2d, y],  # Laszakovits & MacKay 2D model
          ['3D Hybrid', sfc.hybrid_areas, columns_3d, y],  # Rivas-Ubach & Laszakovits-MacKay 3D model
          ['Hybrid (complete)', sfc.hybrid_areas_complete, columns_multi, y],  # Full multi-dimensional
          # Rivas-Ubach definitions (with merged phytochemical category):
          ['Zero rule (R-U)', y_ru],
          ['3D MSCC (R-U)', sfc.rivasubach_areas, columns_3d, y_ru],
          ['MSCC (R-U)', sfc.rivasubach_areas_complete, columns_multi, y_ru],]

for combo in combos:

    lit_acc_comparison['lit_model'].append(combo[0])

    # Generate predictions: use molecclass for literature models, most frequent class for zero rule
    y_pred = sfc.molecclass(df, combo[1], combo[2]) if 'Zero rule' not in combo[0] \
             else [np.unique(combo[-1])[np.argmax([len(np.where(combo[-1] == cat)[0]) for cat in np.unique(combo[-1])])]] * len(combo[-1])

    # get confusion matrix
    confusion_matrices.append([confusion_matrix(y_true=combo[-1],y_pred=y_pred,normalize='true'),
                               np.unique(combo[-1]),combo[0]])
 
    # Calculate both standard and balanced accuracy
    combos2 = [['accuracy', accuracy_score, dict(normalize=True)],
               ['balanced_accuracy', balanced_accuracy_score, dict(sample_weight=weights if weighted else None)]]

    for combo2 in combos2:
        lit_acc_comparison[combo2[0]].append(combo2[1](y_true=combo[-1],
                                                       y_pred=y_pred,
                                                       **combo2[2]))

# Save results to CSV
lit_acc_comparison_df = pd.DataFrame(lit_acc_comparison)
lit_acc_comparison_df.to_csv(f'{csv_folder}/lit_model_comparison.csv', index=False)
print(lit_acc_comparison_df)


for conf_matrix in confusion_matrices:
    # conf_matrix[0] is the matrix, conf_matrix[1] is a list of unique categories
    if len(conf_matrix[0]) > len(conf_matrix[1]):
        conf_matrix[1] = np.append(conf_matrix[1],'unassigned')
    fig, ax = draw_confusion_matrix(conf_matrix[0],conf_matrix[1])

    fig.savefig(f'{comparison_plots_output_folder}/confusion_matrices/absolute_{conf_matrix[2].replace(' ','_').lower()}_confusion_matrix.png',
                dpi=400, facecolor='#fff', bbox_inches='tight')

    plt.close()

