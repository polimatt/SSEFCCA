import numpy as np
import matplotlib.pyplot as plt

# Global variables ------------------------------------------------------------------------------------
weighted = True

vk_region_colours = {
    'lipid': '#F5C308',
    'peptide': '#BD3A53',

    'carbohydrate': '#0040FF',
    'amino_sugar': '#7C9EBF',
    
    'lignin': '#33251E',
    'tannin': '#CE833B',

    'pah': "#808080",

    'unassigned': '#e6e6e6'
}

svc_poly_param_2d = dict(C=10, degree=1, gamma=1) # chosen SVC polynomial kernel parameters after optimisation
svc_poly_param_3d = dict(C=500, degree=3, gamma=0.1)
svc_rbf_param_2d = dict(C=1000, gamma=0.01) # chosen SVC RBF kernel parameters after optimisation
svc_rbf_param_3d = dict(C=1000, gamma=0.01)

svc_poly_param_2d_with_PAHs = dict(C=10, degree=1, gamma='auto')
svc_poly_param_3d_with_PAHs = dict(C=500, degree=3, gamma=0.1)
svc_rbf_param_2d_with_PAHs = dict(C=100, gamma=1)
svc_rbf_param_3d_with_PAHs = dict(C=100, gamma='auto')

# van Krevelen diagram regions definitions --------------------------------------------------------

rivasubach_areas = { # van Krevelen diagram regions from Rivas-Ubach et al. 2018
                'carbohydrate':    {'O/C': [0.8,50],     'H/C': [1.65,2.7],   'N/C': [-1,0]        }, #[[O/C_min,O/C_max],[H/C_min,H/C_max]]
                'lipid':           {'O/C': [-1,0.6],     'H/C': [1.32,50],    'N/C': [-1,0.126]    }, # -1 indicates is used to include 0; 50 is used to include +infinity
                'phytochemical':   {'O/C': [-1,1.15],    'H/C': [-1,1.32],    'N/C': [-1,0.2]      },
                'amino_sugar':     {'O/C': [0.61,50],    'H/C': [1.45,50],    'N/C': [0.07,0.2]    }, #a third item to indicate that this class contains N (put N/C ratio)
                'peptide1':        {'O/C': [0.12,0.6],   'H/C': [0.9,2.5],    'N/C': [0.126,0.7]   },
                'peptide2':        {'O/C': [0.6,1],      'H/C': [1.2,2.5],    'N/C': [0.2,0.7]     },
}

laszakovits_mackay_areas = { # van Krevelen diagram regions from Laszakovits and MacKay 2022
                'amino_sugar':     {'O/C': [0.5,0.91],   'H/C': [1.69,2.33],   },
                'carbohydrate':    {'O/C': [0.55,1.17],  'H/C': [1.43,2]       },
                'lipid':           {'O/C': [0.02,0.37],  'H/C': [1.16,2.33]    },
                'lignin':          {'O/C': [0.11,0.42],  'H/C': [0.77,1.33]    },
                'tannin':          {'O/C': [0.13,0.81],  'H/C': [0.68,1.18]    },
                'peptide':         {'O/C': [0.09,0.75],  'H/C': [0.91,2]       },
}

# adjust Rivas-Ubach et al. 2018 categories to match the categories in our dataset by replacing 'phytochemical' with 'lignin' and 'tannin' based on O/C and H/C ratios from Laszakovits and MacKay 2022
ru_lm_areas = { # van Krevelen diagram regions from Rivas-Ubach et al. 2018
                'carbohydrate':    {'O/C': [0.8,50],      'H/C': [1.65,2.7],   'N/C': [-1,0]        }, #[[O/C_min,O/C_max],[H/C_min,H/C_max]]
                'lipid':           {'O/C': [-1,0.6],      'H/C': [1.32,50],    'N/C': [-1,0.126]    }, # -1 indicates is used to include 0; 50 is used to include +infinity
                'tannin':          {'O/C': [0.13,0.81],   'H/C': [0.68,1.18],  'N/C': [-1,0]        },
                'lignin':          {'O/C': [0.11,0.42],   'H/C': [0.77,1.33],  'N/C': [-1,0]        },
                'amino_sugar':     {'O/C': [0.61,50],     'H/C': [1.45,50],    'N/C': [0.07,0.2]    }, #a third item to indicate that this class contains N (put N/C ratio)
                'peptide1':        {'O/C': [0.12,0.6],    'H/C': [0.9,2.5],    'N/C': [0.126,0.7]   },
                'peptide2':        {'O/C': [0.6,1],       'H/C': [1.2,2.5],    'N/C': [0.2,0.7]     },
}

# Functions -------------------------------------------------------------------------------------------
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


def draw_decision_boundary_plot(estimator, X, categories, xlabel=None, ylabel=None, title=None, savepath=None,
                                level_step=0.1,grid_resolution=500,colors_dict=None):
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
   
    ax.legend(framealpha=1,bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0,fontsize=12)

    ax.set_xlim(0,2.5)
    ax.set_ylim(0,2.5)

    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)

    if title: ax.set_title(title, fontsize=16)

    if savepath: fig.savefig(savepath, dpi = 600, facecolor = '#fff', bbox_inches='tight')


def load_model_onnx(filepath):
    ''''Load an ONNX model from a file and return an InferenceSession.'''

    from onnxruntime import InferenceSession

    with open(filepath, "rb") as f:
        onnx = f.read()

    sess = InferenceSession(onnx, providers=["CPUExecutionProvider"])

    return sess


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

    ax.set_xticks(np.arange(1,len(labels)+1), labels, rotation=xlabel_rotation, ha=ha, fontsize=12)

    ax.set_ylabel(ylabel, fontsize=12)
    
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

    if title: ax.set_title(title, fontsize=14)
    if savepath: fig.savefig(savepath, dpi=600, facecolor='#fff', bbox_inches='tight')

    return fig, ax