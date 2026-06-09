# Author: Mattia Poli

# Shared utilities and constants for MLACCA notebooks.

# This module centralises reusable plotting helpers, model utilities, and
# van Krevelen region definitions so that multiple notebooks can share the
# same logic and settings.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pickle import load

# Global variables ------------------------------------------------------------------------------------
# Styling defaults for plots
axis_label_size = 14
legend_label_size = 12
title_label_size = 16

# Global toggle for weighted training/metrics in helper functions
weighted = True

# Consistent color mapping for molecular categories
vk_region_colours = {
    'lipid': '#F5C308',
    'peptide': '#BD3A53',

    'carbohydrate': '#0040FF',
    'amino_sugar': '#7C9EBF',
    
    'lignin': '#33251E',
    'tannin': '#CE833B',

    'pah': '#808080',

    'phytochemical': '#52D66E',

    'unassigned': '#e6e6e6'
}


hyperparameters_file = 'data/hyperparameters/hyperparameters.pkl'
try:
    with open(hyperparameters_file, 'rb') as f:
        hyperparameters = load(f)
except FileNotFoundError:
    import warnings
    warnings.warn('Hyperparameters file not found.')
    hyperparameters = None

if hyperparameters is not None:
    # KNN hyperparameters
    knn_param_3d = hyperparameters['knn_3d']
    knn_param_3d_with_PAHs = hyperparameters['knn_pahs_3d']

    # SVM (RBF) hyperparameters
    svm_rbf_param_3d = hyperparameters['rbf_svm_3d']
    svm_rbf_param_3d_with_PAHs = hyperparameters['rbf_svm_pahs_3d']

    # SVM (polynomial) hyperparameters
    svm_poly_param_3d = hyperparameters['poly_svm_3d']
    svm_poly_param_3d_with_PAHs = hyperparameters['poly_svm_pahs_3d']


# van Krevelen diagram regions definitions --------------------------------------------------------
# Default dimensionality for 3D van Krevelen space

columns_3d = ['O/C','H/C','N/C']

# van Krevelen diagram regions from Rivas-Ubach et al. 2018
rivasubach_areas_complete = {
                'lipid':           {'O/C': [-np.inf,0.6],   'H/C': [1.32,np.inf],   'N/C': [-np.inf,0.126], 'P/C': [-np.inf,0.35],    'N/P': [-np.inf,5],   },
                'peptide1':        {'O/C': [0.12,0.6],      'H/C': [0.9,2.5],       'N/C': [0.126,0.7],     'P/C': [-np.inf,0.17],    'N/P': None,          },
                'peptide2':        {'O/C': [0.6,1],         'H/C': [1.2,2.5],       'N/C': [0.2,0.7],       'P/C': [-np.inf,0.17],    'N/P': None,          },
                'amino_sugar':     {'O/C': [0.61,np.inf],   'H/C': [1.45,np.inf],   'N/C': [0.07,0.2],      'P/C': [-np.inf,0.3],     'N/P': [-np.inf,2],   },
                'carbohydrate':    {'O/C': [0.8,np.inf],    'H/C': [1.65,2.7],      'N/C': [0,0],           'P/C': None,              'N/P': None,          },
                'phytochemical':   {'O/C': [-np.inf,1.15],  'H/C': [-np.inf,1.32],  'N/C': [-np.inf,0.2],   'P/C': [-np.inf,0.2],     'N/P': [-np.inf,3],   },
}

# Subset of Rivas-Ubach regions for 3D space (O/C, H/C, N/C)
rivasubach_areas = {x:{y:rivasubach_areas_complete[x][y] for y in rivasubach_areas_complete[x] if y in columns_3d} for x in rivasubach_areas_complete}

# van Krevelen diagram regions from Laszakovits and MacKay 2022
laszakovits_mackay_areas = {
                'amino_sugar':     {'O/C': [0.56,0.95],  'H/C': [1.62,2.35]    },
                'carbohydrate':    {'O/C': [0.56,1.23],  'H/C': [1.53,2.2]     },
                'lignin':          {'O/C': [0.21,0.44],  'H/C': [0.86,1.34]    },
                'lipid':           {'O/C': [0.01,0.35],  'H/C': [1.34,2.18]    },
                'peptide':         {'O/C': [0.17,0.48],  'H/C': [1.33,1.84]    },
                'tannin':          {'O/C': [0.16,0.84],  'H/C': [0.7,1.01]     },
}

# Adjust Rivas-Ubach et al. 2018 categories to match our dataset by
# replacing 'phytochemical' with 'lignin' and 'tannin' using O/C and H/C
# bounds from Laszakovits and MacKay 2022.

def adjust_ru_areas(ru_area):
    """
    Replace phytochemical region with lignin/tannin in Rivas-Ubach definitions.
    
    Takes the Rivas-Ubach et al. (2018) van Krevelen region definitions and
    replaces the 'phytochemical' category with separate 'lignin' and 'tannin'
    regions using O/C and H/C boundaries from Laszakovits & MacKay (2022).
    Other dimensions (N/C, P/C, N/P) are inherited from the original
    phytochemical region definition.
    
    Parameters
    ----------
    ru_area : dict
        Dictionary mapping molecular class names to their region boundaries.
        Each class maps to a dict of dimension names (e.g., 'O/C', 'H/C') to
        [min, max] ranges or None.
    
    Returns
    -------
    dict
        Modified region definitions with 'phytochemical' removed and 'lignin'
        and 'tannin' added as separate categories.
    
    Notes
    -----
    This function enables using the more chemically-specific lignin/tannin
    categories instead of the broader phytochemical category, while maintaining
    compatibility with the R-U region framework.
    """
    adjusted_area = {x:ru_area[x] for x in ru_area if x != 'phytochemical'}
    ru_dims = list(ru_area['phytochemical'].keys())
    for cat in ['lignin','tannin']:
        adjusted_area[cat] = laszakovits_mackay_areas[cat].copy()

        for dim in ru_dims:
            if dim not in adjusted_area[cat].keys():
                adjusted_area[cat][dim] = ru_area['phytochemical'][dim]

    return adjusted_area

# Adjusted RU regions compatible with the dataset categories
ru_lm_areas_complete = adjust_ru_areas(rivasubach_areas_complete)
ru_lm_areas = adjust_ru_areas(rivasubach_areas)

# Functions -------------------------------------------------------------------------------------------

def set_axis_ticks(data, ax, axis:str='y', major_ticks_interval=0.1, minor_ticks_interval=0.05, rounding=2,
                   lower_lim=None, upper_lim=None):
    """
    Set axis limits and ticks with automatic bounds calculation and sensible rounding.
    
    Automatically determines appropriate axis limits based on data range, with
    optional override. Applies major and minor ticks at specified intervals.
    Works with standard 2D plots (x, y axes) and 3D plots (x, y, z axes).
    
    Parameters
    ----------
    data : array-like, dict, or pd.DataFrame
        Data values used to determine axis limits. If dict or DataFrame,
        values are extracted automatically.
    ax : matplotlib.axes.Axes
        The axes object to modify.
    axis : {'x', 'y', 'z'}, default 'y'
        Which axis to configure.
    major_ticks_interval : float, default 0.1
        Spacing between major tick marks.
    minor_ticks_interval : float or None, default 0.05
        Spacing between minor tick marks. If None, minor ticks are not set.
    rounding : int, default 2
        Decimal places for rounding axis limits.
    lower_lim : float or None, optional
        Manual override for lower axis limit. If None, computed from data.
    upper_lim : float or None, optional
        Manual override for upper axis limit. If None, computed from data.
    """
    import matplotlib as mpl

    # Allow dicts/DataFrames to be passed directly
    if type(data) in [dict, pd.DataFrame]:
        data = data.values()

    lim = (lower_lim if lower_lim is not None else np.round(np.floor(10**rounding*np.min(list(data)))/10**rounding,rounding)-10**-rounding,
           upper_lim if upper_lim is not None else np.round(np.ceil(10**rounding*np.max(list(data)))/10**rounding,rounding)+10**-rounding)
    
    # Pick the correct axis setters
    if axis == 'y':
        set_axislim = ax.set_ylim
        set_axisticks = ax.set_yticks
        minor_ticks = ax.yaxis.set_minor_locator
    elif axis == 'x':
        set_axislim = ax.set_xlim
        set_axisticks = ax.set_xticks
        minor_ticks = ax.xaxis.set_minor_locator
    elif axis == 'z':
        set_axislim = ax.set_zlim
        set_axisticks = ax.set_zticks
        minor_ticks = ax.zaxis.set_minor_locator

    # Apply limits and ticks
    set_axislim(lim)
    set_axisticks(np.arange(np.round(np.floor(1e2*lim[0])/1e2,1),lim[1]+0.01, major_ticks_interval))

    ax.tick_params(axis=axis, which='minor')
    if minor_ticks_interval is not None: minor_ticks(mpl.ticker.MultipleLocator(minor_ticks_interval))


def clf_pipeline(clf, calibrated=True):
    """
    Build a classifier pipeline with preprocessing.
    
    Creates a scikit-learn pipeline that includes:
    1. StandardScaler for feature normalization
    2. Classifier with specified parameters
    3. Optional CalibratedClassifierCV wrapper for probability calibration
    
    Parameters
    ----------
    clf
        SKL classifier .
    calibrated : bool, default True
        If True, wraps clf in CalibratedClassifierCV to improve probability
        estimates using cross-validation.
    
    Returns
    -------
    sklearn.pipeline.Pipeline
        A fitted pipeline ready for training with .fit(X, y).
    
    Notes
    -----
    StandardScaler is essential for distance-based algorithms
    sensitive to feature scales. Calibration improves probability estimates
    which are useful for decision boundary visualization.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.calibration import CalibratedClassifierCV

    return Pipeline(steps=[("scaler", StandardScaler()),
                           ("clf", CalibratedClassifierCV(clf) if calibrated else clf)])


def get_weights_array(y):
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    weights_dict = {cls: weight for cls, weight in zip(np.unique(y), class_weights)}
    weights_arr = np.array([weights_dict[cat] for cat in y])
    return weights_arr

def train_test(X, y, train_size=0.8, weighted_yn=weighted, random_state=None, stratify_yn=False):
    """
    Split data into train/test sets with optional class weight computation.
    
    Performs stratified train-test split and optionally computes balanced
    class weights for both splits. Weights are useful for handling class
    imbalance in training and evaluation metrics.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    y : array-like of shape (n_samples,)
        Target class labels.
    train_size : float, default 0.8
        Proportion of dataset to include in training split (0.0 to 1.0).
    weighted_yn : bool, default weighted (global variable)
        If True, compute balanced class weights for both train and test sets.
        Weights are inversely proportional to class frequencies.
    random_state : int or None, optional
        Random seed for reproducible splits. If None, split is random.
    
    Returns
    -------
    X_train : ndarray of shape (n_train_samples, n_features)
        Training features.
    X_test : ndarray of shape (n_test_samples, n_features)
        Test features.
    y_train : ndarray of shape (n_train_samples,)
        Training labels.
    y_test : ndarray of shape (n_test_samples,)
        Test labels.
    weights_train : ndarray of shape (n_train_samples,) or None
        Sample weights for training set. None if weighted_yn is False.
    weights_test : ndarray of shape (n_test_samples,) or None
        Sample weights for test set. None if weighted_yn is False.
    """

    from sklearn.model_selection import train_test_split

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=random_state,
                                                        stratify=None if not stratify_yn else y)
    
    if weighted_yn:
        # Compute balanced weights for each split to handle class imbalance
        weights_train = get_weights_array(y_train)
        weights_test = get_weights_array(y_test)
    
    else:
        weights_train = None
        weights_test = None

    return X_train, X_test, y_train, y_test, weights_train, weights_test


def molecclass(df: pd.DataFrame, areas: dict, dims=columns_3d) -> np.ndarray:
    """
    Assign molecular classes based on van Krevelen region boundaries.
    
    Classifies compounds by checking if their elemental ratios
    fall within predefined van Krevelen diagram regions. Each compound is
    assigned to the first matching region. Compounds outside all regions are
    labeled 'unassigned'.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing elemental ratio columns. Must include all
        dimensions specified in 'dims' parameter.
    areas : dict
        Dictionary mapping molecular class names to region boundaries.
        Structure: {class_name: {dimension: [min, max] or None, ...}}
        Example: {'lipid': {'O/C': [0, 0.6], 'H/C': [1.32, np.inf], ...}}
        Dimensions with None are ignored for that class.
    dims : list of str, default columns_3d (['O/C', 'H/C', 'N/C'])
        Column names representing dimensions to check. Must be present in df.
    
    Returns
    -------
    np.ndarray of shape (n_samples,)
        Array of assigned class labels. One label per row in df.
        Labels match keys in 'areas' dict, or 'unassigned' for no match.
    
    Notes
    -----
    - Classification order matters: first matching region wins
    - Special handling: 'peptide1' and 'peptide2' are merged to 'peptide'
    - Boundaries are inclusive: min <= value <= max
    - Multi-dimensional: ALL specified dimensions must fall within bounds
    """

    df_copy = df.copy()
    df_copy.reset_index(drop=True, inplace=True)

    assignments = ['unassigned'] * len(df_copy)

    for i in df_copy[dims].index:
        ratios = df_copy[dims].loc[i]

        for a in areas:
            counter = 0

            # Check each dimension against class boundaries
            notNone_dims = [dim for dim in dims if areas[a][dim] is not None]
            for d in notNone_dims:
                if ratios[d] >= np.min(areas[a][d]) and ratios[d] <= np.max(areas[a][d]):
                    counter += 1

            if counter == len(notNone_dims):
                assignments[i] = a
                break
    
    assignments = np.array(assignments)
    # Merge peptide sub-classes into a single label
    assignments[[a in ['peptide1','peptide2'] for a in assignments]] = 'peptide'

    return assignments


def class_wise_accuracy(y_true:np.ndarray|list, y_pred:np.ndarray|list) -> dict:
    from sklearn.metrics import accuracy_score
    """
    Calculate class-wise accuracy for a multi-class classification problem.
    
    Parameters:
    -----------
    y_true : np.ndarray | list
        True class labels
    y_pred : np.ndarray | list
        Predicted class labels
    
    Returns:
    --------
    dict : Dictionary with class labels as keys and their corresponding accuracies as values
    """
    accuracy_dict = {}
    if type(y_true) == list:
        y_true = np.array(y_true)
    if type(y_pred) == list:
        y_pred = np.array(y_pred)
    
    for class_ in np.unique(y_true):
        idx = np.where(y_true == class_)[0]
        cls_accuracy = accuracy_score(y_true[idx], y_pred[idx])
        accuracy_dict[class_] = cls_accuracy

    return accuracy_dict


def draw_decision_boundary_plot(estimator, X, categories, xlabel=None, ylabel=None, title=None, savepath=None,
                                level_step=0.1, grid_resolution=500, colors_dict=None, xlim=(0, 2.5), ylim=(0, 2.5)):
    """
    Plot calibrated decision boundaries for 2D feature spaces with probability contours.
    
    Creates a contour plot showing class prediction probabilities across a 2D
    feature space. Uses a trained classifier's predict_proba method to visualize
    decision boundaries. Optionally displays probability levels as contour lines.
    
    Parameters
    ----------
    estimator : sklearn estimator
        Trained classifier with predict_proba method (e.g., calibrated KNN, SVM).
    X : array-like of shape (n_samples, 2)
        2D feature data used to determine plot bounds.
    categories : array-like of str
        Ordered list of class names matching estimator's classes_.
    xlabel : str, optional
        Label for x-axis (e.g., 'O/C').
    ylabel : str, optional
        Label for y-axis (e.g., 'H/C').
    title : str, optional
        Plot title.
    savepath : str, optional
        File path to save figure (e.g., 'output/plot.png').
        If None, figure is not saved.
    level_step : float or None, default 0.1
        Probability interval between contour levels (0.0 to 1.0).
        If None, no contour lines are drawn.
    grid_resolution : int, default 500
        Number of grid points per axis for boundary resolution.
        Higher values give smoother boundaries but slower plotting.
    colors_dict : dict, optional
        Mapping of category names to color codes.
        Example: {'lipid': '#F5C308', 'peptide': '#BD3A53'}
    xlim : tuple of float, default (0, 2.5)
        X-axis limits as (min, max).
    ylim : tuple of float, default (0, 2.5)
        Y-axis limits as (min, max).
    
    Returns
    -------
    fig : plt.Figure
        The figure object.
    ax : plt.Axes
        The axes object.
    
    Notes
    -----
    - Requires estimator to have predict_proba method (use calibration if needed)
    - Legend shows both class colors and probability level indicators
    - Figure is saved at 600 DPI with white background
    """
    from sklearn.inspection import DecisionBoundaryDisplay

    fig, ax = plt.subplots()

    # Levels for contour plots (probability thresholds)
    levels = np.arange(0, 1 + level_step, level_step) if level_step else None

    disp = DecisionBoundaryDisplay.from_estimator(
        estimator,
        X,
        response_method="predict_proba",
        plot_method="contourf", # contourf, contour, pcolormesh
        ax=ax,
        grid_resolution=grid_resolution,
        levels=levels,
        multiclass_colors=[colors_dict[cat] for cat in categories] if colors_dict else None
    )

    # for cs in disp.surface_:
    #     plt.clabel(cs, inline=False, fontsize=8,colors='#666')

    # Build a clean legend with category names
    for cat in categories:
        cat_mod = cat.replace('_',' ').capitalize() + '-like'
        if 'Pah' in cat_mod:
            cat_mod = 'PAH-like'
        ax.scatter(None,None,zorder=1,alpha=1,label=cat_mod,c=colors_dict[cat] if colors_dict else None)

    if level_step:
        for level in np.arange(level_step, 1 + level_step, level_step):
            ax.plot([None,None],[None,None],
                    c='k',alpha=level,lw=10,label=f'{(100*level):.0f}% probability')
   
    ax.legend(framealpha=1,bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0,fontsize=legend_label_size)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.set_xlabel(xlabel, fontsize=axis_label_size)
    ax.set_ylabel(ylabel, fontsize=axis_label_size)

    if title: ax.set_title(title, fontsize=title_label_size)

    if savepath: fig.savefig(savepath, dpi = 600, facecolor = '#fff', bbox_inches='tight')

    return fig, ax


def draw_boxplot(data_dict, ylabel, title=None, savepath=None, colours=[], hline=None, # ylim=None,
                 xlabel_rotation=45, ha='right', figsize=None):
    """
    Create a customizable boxplot with optional colors and reference lines.
    
    Generates a matplotlib boxplot from dictionary data with support for
    custom box colors, horizontal reference lines, rotated labels, and
    automatic saving.
    
    Parameters
    ----------
    data_dict : dict
        Dictionary mapping category names (keys) to data arrays (values).
        Each key becomes an x-axis label, each value is plotted as a box.
        Example: {'Class A': [1, 2, 3], 'Class B': [2, 3, 4]}
    ylabel : str
        Label for y-axis.
    title : str, optional
        Plot title.
    savepath : str, optional
        File path to save figure. If None, figure is not saved.
    colours : list of str, default []
        Color codes for box faces. If single color provided, applied to all.
        If multiple colors, must match number of boxes.
        If empty list, default matplotlib colors are used.
    hline : float, list, or tuple, optional
        Y-coordinate(s) for horizontal reference line(s).
        Single value or sequence of values. Lines drawn in red dashed style.
    xlabel_rotation : float, default 45
        Rotation angle for x-axis labels in degrees.
    ha : {'left', 'center', 'right'}, default 'right'
        Horizontal alignment for x-axis labels.
    figsize : tuple of float, optional
        Figure size as (width, height) in inches.
        If None, uses matplotlib default.
    
    Returns
    -------
    fig : plt.Figure
        The figure object.
    ax : plt.Axes
        The axes object.
    
    Notes
    -----
    - Medians are shown as black solid lines
    - Outliers are displayed by default
    - Figure is saved at 600 DPI with white background if savepath provided
    """
    fig, ax = plt.subplots(figsize=figsize)

    labels = list(data_dict.keys())
    data = list(data_dict.values())

    bplot = ax.boxplot(data,
                       medianprops = dict(color='k',ls='-'),
                       patch_artist = True if len(colours)>0 else False,
                       showfliers = True)

    ax.set_xticks(np.arange(1,len(labels)+1), labels, rotation=xlabel_rotation, ha=ha, fontsize=axis_label_size)

    ax.set_ylabel(ylabel, fontsize=axis_label_size)
    
    # Apply custom box colors if provided
    if len(colours)>0:
        if len(colours)==1:
            colours = colours * len(labels)
        assert len(colours)==len(labels), "Number of colours must match number of boxplots"
        for patch, colour in zip(bplot['boxes'], colours):
            patch.set_facecolor(colour)

    # Optional horizontal reference line(s)
    if hline:
        if type(hline) in [list, tuple]:
            for hl in hline:
                ax.axhline(y=hl, color='r', linestyle='--', lw=1)
        else:
            ax.axhline(y=hline, color='r', linestyle='--', lw=1)

    if title: ax.set_title(title, fontsize=title_label_size)
    if savepath: fig.savefig(savepath, dpi=600, facecolor='#fff', bbox_inches='tight')

    return fig, ax


def Shannon_diversity_index(labels:np.ndarray|pd.Series|list) -> float:
    label_copy = labels if isinstance(labels,np.ndarray) else np.array(labels)

    counts = [len([y for y in label_copy==x if y]) for x in np.unique(label_copy)]

    counts_sum = len(label_copy)
    P = [c/counts_sum for c in counts]

    return -np.sum([p_i * np.log(p_i) for p_i in P])