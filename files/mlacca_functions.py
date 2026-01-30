import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Global variables ------------------------------------------------------------------------------------
axis_label_size = 14
legend_label_size = 12
title_label_size = 16

weighted = True

vk_region_colours = {
    'lipid': '#F5C308',
    'peptide': '#BD3A53',

    'carbohydrate': '#0040FF',
    'amino_sugar': '#7C9EBF',
    
    'lignin': '#33251E',
    'tannin': '#CE833B',

    'pah': "#808080",

    'phytochemical': '#52D66E',

    'unassigned': '#e6e6e6'
}

knn_param_2d = dict(n_neighbors=50, weights='distance')
knn_param_3d = dict(n_neighbors=10, weights='distance')
knn_param_2d_with_PAHs = dict(n_neighbors=10, weights='distance')
knn_param_3d_with_PAHs = dict(n_neighbors=10, weights='distance')

svm_rbf_param_2d = dict(C=100, gamma='auto') # chosen SVC RBF kernel parameters after optimisation
svm_rbf_param_3d = dict(C=100, gamma='auto')
svm_rbf_param_2d_with_PAHs = dict(C=100, gamma='auto')
svm_rbf_param_3d_with_PAHs = dict(C=100, gamma='auto')

svm_poly_param_2d = dict(C=10, degree=1, gamma=1) # chosen SVC polynomial kernel parameters after optimisation
svm_poly_param_3d = dict(C=100, degree=5, gamma='auto')
svm_poly_param_2d_with_PAHs = dict(C=10, degree=2, gamma='auto')
svm_poly_param_3d_with_PAHs = dict(C=10, degree=5, gamma='auto')


# van Krevelen diagram regions definitions --------------------------------------------------------

rivasubach_areas = { # van Krevelen diagram regions from Rivas-Ubach et al. 2018
                'carbohydrate':    {'O/C': [0.8,50],     'H/C': [1.65,2.7],   'N/C': [-1,0]        }, #[[O/C_min,O/C_max],[H/C_min,H/C_max]]
                'lipid':           {'O/C': [-1,0.6],     'H/C': [1.32,50],    'N/C': [-1,0.126]    }, # -1 indicates is used to include 0; 50 is used to include +infinity
                'phytochemical':   {'O/C': [-1,1.15],    'H/C': [-1,1.32],    'N/C': [-1,0.2]      },
                'amino_sugar':     {'O/C': [0.61,50],    'H/C': [1.45,50],    'N/C': [0.07,0.2]    }, #a third item to indicate that this class contains N (put N/C ratio)
                'peptide1':        {'O/C': [0.12,0.6],   'H/C': [0.9,2.5],    'N/C': [0.126,0.7]   },
                'peptide2':        {'O/C': [0.6,1],      'H/C': [1.2,2.5],    'N/C': [0.2,0.7]     },
}

# laszakovits_mackay_areas = { # van Krevelen diagram regions from Laszakovits and MacKay 2022
#                 'amino_sugar':     {'O/C': [0.5,0.91],   'H/C': [1.69,2.33],   },
#                 'carbohydrate':    {'O/C': [0.55,1.17],  'H/C': [1.43,2]       },
#                 'peptide':         {'O/C': [0.09,0.75],  'H/C': [0.91,2]       },
#                 'lipid':           {'O/C': [0.02,0.37],  'H/C': [1.16,2.33]    },
#                 'lignin':          {'O/C': [0.11,0.42],  'H/C': [0.77,1.33]    },
#                 'tannin':          {'O/C': [0.13,0.81],  'H/C': [0.68,1.18]    },
# }

laszakovits_mackay_areas = { # van Krevelen diagram regions from Laszakovits and MacKay 2022
                'amino_sugar':     {'O/C': [0.56,0.95],  'H/C': [1.62,2.35],   },
                'carbohydrate':    {'O/C': [0.56,1.23],  'H/C': [1.53,2.2]     },
                'peptide':         {'O/C': [0.17,0.48],  'H/C': [1.33,1.84]    },
                'lipid':           {'O/C': [0.01,0.35],  'H/C': [1.34,2.18]    },
                'lignin':          {'O/C': [0.21,0.44],  'H/C': [0.86,1.34]    },
                'tannin':          {'O/C': [0.16,0.84],  'H/C': [0.7,1.01]     },
}

# adjust Rivas-Ubach et al. 2018 categories to match the categories in our dataset by replacing 'phytochemical' with 'lignin' and 'tannin' based on O/C and H/C ratios from Laszakovits and MacKay 2022
ru_lm_areas = {x:rivasubach_areas[x] for x in rivasubach_areas if x != 'phytochemical'}
for cat in ['lignin','tannin']:
    ru_lm_areas[cat] = laszakovits_mackay_areas[cat]
    ru_lm_areas[cat]['N/C'] = rivasubach_areas['phytochemical']['N/C']

# Functions -------------------------------------------------------------------------------------------

def set_axis_ticks(data,ax,axis:str='y',major_ticks_interval=0.1,minor_ticks_interval=0.05,rounding=2,
                   lower_lim=None,upper_lim=None):
    import matplotlib as mpl

    if type(data) in [dict,pd.DataFrame]: data = data.values()

    lim = (lower_lim if lower_lim is not None else np.round(np.floor(10**rounding*np.min(list(data)))/10**rounding,rounding)-10**-rounding,
           upper_lim if upper_lim is not None else np.round(np.ceil(10**rounding*np.max(list(data)))/10**rounding,rounding)+10**-rounding)
    
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

    set_axislim(lim)
    set_axisticks(np.arange(np.round(np.floor(1e2*lim[0])/1e2,1),lim[1]+0.01, major_ticks_interval))

    ax.tick_params(axis=axis, which='minor')
    if minor_ticks_interval is not None: minor_ticks(mpl.ticker.MultipleLocator(minor_ticks_interval))


def knn_pipeline(KNeighborsClassifier_parameters,calibrated=True):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    knn_clf = KNeighborsClassifier(**KNeighborsClassifier_parameters)
    if calibrated:
        from sklearn.calibration import CalibratedClassifierCV
        knn_clf = CalibratedClassifierCV(knn_clf)

    return Pipeline(steps=[("scaler", StandardScaler()), ("knn", knn_clf)])


def train_test(X, y,train_size=0.8,weighted_yn=weighted,random_state=5):

    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_class_weight

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=random_state)

    if weighted_yn:
        class_weights_train = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weights_test = compute_class_weight('balanced', classes=np.unique(y_test), y=y_test)

        weights_dict_train = {cls: weight for cls, weight in zip(np.unique(y_train), class_weights_train)}
        weights_dict_test = {cls: weight for cls, weight in zip(np.unique(y_test), class_weights_test)}

        weights_train = np.array([weights_dict_train[cat] for cat in y_train])
        weights_test = np.array([weights_dict_test[cat] for cat in y_test])
    
    else:
        weights_train = None
        weights_test = None

    return X_train, X_test, y_train, y_test, weights_train, weights_test


def molecclass(df:pd.DataFrame,areas:dict,dims=['O/C','H/C','N/C']) ->  np.ndarray:

    assignments = ['unassigned'] * len(df)

    for i in df[dims].index:
        ratios = df[dims].loc[i]

        for a in areas:
            counter = 0

            for d in dims:
                if ratios[d] >= np.min(areas[a][d]) and ratios[d] <= np.max(areas[a][d]):
                    counter += 1
        
            if counter == len(dims):
                assignments[i] = a
                break
    
    assignments = np.array(assignments)
    assignments[np.where(assignments == 'peptide1')] = 'peptide'
    assignments[np.where(assignments == 'peptide2')] = 'peptide'

    return assignments


def draw_decision_boundary_plot(estimator, X, categories, xlabel=None, ylabel=None, title=None, savepath=None,
                                level_step=0.1,grid_resolution=500,colors_dict=None,xlim=(0,2.5),ylim=(0,2.5)):
    from sklearn.inspection import DecisionBoundaryDisplay

    fig, ax = plt.subplots()

    levels = np.arange(0, 1 + level_step, level_step) if level_step else None # levels for contour plots

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


def load_pickle_model(filepath:str):
    from pickle import load
    with open(filepath, "rb") as f:
        clf = load(f)
    return clf


def draw_boxplot(data_dict, ylabel, title=None, savepath=None,colours=[],hline=None,#ylim=None,
                 xlabel_rotation=45,ha='right',figsize=None):
    fig, ax = plt.subplots(figsize=figsize)

    labels = list(data_dict.keys())
    data = list(data_dict.values())

    bplot = ax.boxplot(data,
                       medianprops = dict(color='k',ls='-'),
                       patch_artist = True if len(colours)>0 else False,
                       showfliers = True)

    ax.set_xticks(np.arange(1,len(labels)+1), labels, rotation=xlabel_rotation, ha=ha, fontsize=axis_label_size)

    ax.set_ylabel(ylabel, fontsize=axis_label_size)
    
    if len(colours)>0:
        if len(colours)==1:
            colours = colours * len(labels)
        assert len(colours)==len(labels), "Number of colours must match number of boxplots"
        for patch, colour in zip(bplot['boxes'], colours):
            patch.set_facecolor(colour)

    if hline:
        if type(hline) in [list, tuple]:
            for hl in hline:
                ax.axhline(y=hl, color='r', linestyle='--', lw=1)
        else:
            ax.axhline(y=hline, color='r', linestyle='--', lw=1)

    if title: ax.set_title(title, fontsize=title_label_size)
    if savepath: fig.savefig(savepath, dpi=600, facecolor='#fff', bbox_inches='tight')

    return fig, ax