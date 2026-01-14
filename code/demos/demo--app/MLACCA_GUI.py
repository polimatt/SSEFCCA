import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.backend_bases import key_press_handler
import plotly.graph_objects as go
import os
from sklearn.calibration import CalibratedClassifierCV
import pickle as pkl

from tkinter import *
from tkinter import filedialog
from tkinter import messagebox

root = Tk()
root.title('MLACCA') #Machine Learning Assisted Compound Class Assignment
root.geometry(f"1000x400")
root.iconbitmap("files_for_app\\logo\\icon.ico")

models_dir_txt = '.\\files_for_app\\models_dir.txt'
f = open(models_dir_txt)
models_dir = f.readline()
f.close() 
models_paths = [x for x in os.listdir(models_dir) if x.endswith('.pkl')]

model_dict = {x:y for x, y in zip([x.replace('.pkl','') for x in models_paths],models_paths)}

model_chosen = StringVar(root)
model_chosen.set(list(model_dict.keys())[0])

plot_2DvK_bool = BooleanVar()


vk_region_colours = {
    'lipid': '#F5C308',
    'peptide': '#BD3A53',
    'carbohydrate': '#0040FF',
    'amino_sugar': '#7C9EBF',
    'lignin': '#33251E',
    'tannin': '#CE833B',
    'pah': "#808080",
}

input_string = 'Directory containing the data to be assigned'
output_string = 'Directory where the assigned data will be saved'

#%%
def browsedatadir():
    global data_dir
    data_dir = filedialog.askdirectory()
    label_data_file_explorer.configure(text=f'{input_string}: {data_dir}')

def browseoutputdir():
    global output_dir
    output_dir = filedialog.askdirectory()
    label_output_file_explorer.configure(text=f'{output_string}: {output_dir}')

def browse3DvKdir():
    global threeDvK_dir
    threeDvK_dir = filedialog.askdirectory()
    label_save3DvK_explorer.configure(text=f'Directory where the 3D van Krevelen diagrams will be saved: {threeDvK_dir}')

def compound_classification(data_df:pd.DataFrame,model:CalibratedClassifierCV,
                            molec_form_column = 'Molecular Formula',
                            ratios_needed:list = ['O/C','H/C','N/C'],
                            isotope_pairs:list[list] = [['C','13C'],['H','2H'],['N','15N'],['O','18O'],['S','34S']]) -> pd.DataFrame:

    assert type(model) == CalibratedClassifierCV, '`model` must be a `skl.calibration.CalibratedClassifierCV`'
    assert molec_form_column in data_df.columns, '`molec_form_column` must be the column of the `data_df` dataframe containing the assigned empirical formulae.'

    # copy the dataframe into another variable the function will be working on
    # substitute all NaNs with zeros and only select the rows with assigned empirical formulae
    assigned_df = data_df.copy()
    assigned_df.fillna(0,inplace=True)
    assigned_df = assigned_df[assigned_df[molec_form_column]!=0]

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
    
    # predict compound classes, their probabilities, and add them as new columns
    predictions = model.predict(assigned_df[ratios_needed].values)
    max_probabilities = np.max(model.predict_proba(assigned_df[ratios_needed].values),axis=1)
    assigned_df['compound_class'] = predictions
    assigned_df['probability'] = max_probabilities

    return assigned_df


# main function
def run_script():

    try: data_dir is None
    except NameError: messagebox.showerror('Error','Please provide a path to a data folder')

    try: output_dir is None
    except NameError: messagebox.showerror('Error','Please provide a path to save the outputs in')

    with open(f'{models_dir}\\{model_dict[model_chosen.get()]}', "rb") as f:
        model = pkl.load(f)

    files_list = [x for x in os.listdir(data_dir) if x.endswith('.csv')]

    if plot_2DvK_bool.get(): dfs = dict()

    for file in files_list:
        file_path = f'{data_dir}\\{file}'
        data_df = pd.read_csv(file_path)
        data_df = compound_classification(data_df,model)
        data_df.to_csv(f'{output_dir}\\{file}')

        if plot_2DvK_bool.get(): dfs[file.replace('.csv','')]=data_df

        try:
            if threeDvK_dir:
                save_3D_vK(data_df,file.replace('.csv',''),f'{threeDvK_dir}\\{file.replace('.csv','.html')}')
        except: pass

    if plot_2DvK_bool.get(): open2DvKwindow(dfs)



def open2DvKwindow(dfs):
    dpi = 600
    twoDvK_window = Toplevel()
    twoDvK_window.title("MLACCA - 2D van Krevelen Diagrams")
    twoDvK_window.iconbitmap("files_for_app\\logo\\icon.ico")
    w2_width = 800
    w2_height = 600+40+50
    twoDvK_window.geometry(f"{w2_width}x{w2_height}")

    file_chosen = StringVar(twoDvK_window)
    file_chosen.set(list(dfs.keys())[0])

    optionmenu_file = OptionMenu(twoDvK_window, file_chosen, *list(dfs.keys()))

    def plot_2DvK():

        chosen_df = dfs[file_chosen.get()]

        fig, ax = plt.subplots(constrained_layout=True, figsize=(4801/dpi, 2927/dpi))

        for compound_class in chosen_df['compound_class'].unique():

            ax.scatter(chosen_df[chosen_df['compound_class']==compound_class]['O/C'],
                    chosen_df[chosen_df['compound_class']==compound_class]['H/C'],
                    c=vk_region_colours[compound_class],
                    label=f'{compound_class.replace('_',' ').capitalize()}-like' if 'pah' not in compound_class else 'PAH-like',
                    alpha=0.8)

        ax.set_xlabel('O/C',fontsize=14)
        ax.set_ylabel('H/C',fontsize=14)
        ax.set_title(f'2D van Krevelen Diagram\nwith Compound Class Assignment for\n{file_chosen.get()}',fontsize=16)
        ax.legend(framealpha=1,bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0,fontsize=12)

        canvas = FigureCanvasTkAgg(fig, master=twoDvK_window)  
        canvas.draw()
        toolbar = NavigationToolbar2Tk(canvas, twoDvK_window, pack_toolbar=False)
        toolbar.update()
        canvas.mpl_connect("key_press_event", lambda event: print(f"you pressed {event.key}"))
        canvas.mpl_connect("key_press_event", key_press_handler)      

        plt.close()

        canvas.get_tk_widget().grid(column=0,row=2,columnspan=5,pady=(10,0))
        toolbar.grid(column=0,row=3,columnspan=5)

    button_plot = Button(twoDvK_window, text='Plot', command=plot_2DvK)

    optionmenu_file.grid(column=3,row=0,padx=w2_width/2,pady=10)
    button_plot.grid(column=3,row=1,padx=w2_width/2,pady=10)



def save_3D_vK(df:pd.DataFrame,name:str,savepath:str):
    vizList = []
    for compound_class in df['compound_class'].unique():

        globals()[f"D_{compound_class}"]=go.Scatter3d(

            x = df[df['compound_class'] == compound_class]['O/C'],
            y = df[df['compound_class'] == compound_class]['H/C'],
            z = df[df['compound_class'] == compound_class]['N/C'],

            name = f'{compound_class.replace('_',' ').capitalize()}-like' if 'pah' not in compound_class else 'PAH-like',

            hovertemplate = 'O/C: %{x} <br>' + \
                            'H/C: %{y} <br>' + \
                            'N/C: %{z} <br>',

            mode='markers',
            marker=dict(
                size=8,
                color=vk_region_colours[compound_class],           
                opacity=0.8,
                symbol='circle'
            )
        )

        vizList.append(globals()[f"D_{compound_class}"]) 

    fig = go.Figure(data=vizList)

    camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=-1, y=-1.75, z=1.5)
        )

    fig.update_layout(
        scene_camera=camera,
        autosize=True,
        template="plotly_white",
        scene={
            "xaxis_title":'O/C',
            "yaxis_title":'H/C',
            "zaxis_title":'N/C'
        },
        width=800, height=600,
        showlegend=True,
        legend_font_size = 14,
        title = f'3D van Krevelen Diagram with Compound Class Assignment for {name}',
        title_font_size = 20
    )
    # 3D vK diagrams created with Plotly are saved as HTML files
    if not savepath.endswith('.html'):
        savepath += '.html'
    fig.write_html(savepath)


#%%
# declare the elements of the GUI

label_data_file_explorer = Label(root,
                                 text = input_string,
                                 fg = "#000")
button_data_explore = Button(root,
                             text = "Browse Data Folder",
                             command = browsedatadir)

optionmenu_model = OptionMenu(root, model_chosen, *list(model_dict.keys()))

label_output_file_explorer = Label(root,
                                   text = output_string,
                                   fg = "#000")

button_output_explore = Button(root,
                               text = "Browse Output Folder",
                               command = browseoutputdir)

checkbutton_plot_2DvK = Checkbutton(root, text = "Plot 2D van Krevelen diagrams",variable = plot_2DvK_bool)

label_save3DvK_explorer = Label(root,text = "If you want to save the 3D van Krevelen diagrams, please select a folder to save them in")

button_save3DvK_explore = Button(root,
                                 text = "Browse Folder to save 3D van Krevelen diagrams in",
                                 command = browse3DvKdir)

submit_button = Button(root, text='Run', command=run_script)

# position all elements within the window

label_data_file_explorer.pack(pady=(5,0))
button_data_explore.pack(pady=(0,5))

Label(root,text = "Select the assignment model").pack(pady=(5,0))

optionmenu_model.pack(pady=(0,5))

label_output_file_explorer.pack(pady=(5,0))
button_output_explore.pack(pady=(0,5))

checkbutton_plot_2DvK.pack(pady=10)
label_save3DvK_explorer.pack(pady=(5,0))
button_save3DvK_explore.pack(pady=(0,5))

submit_button.pack(pady=10)


# run the app
root.mainloop()