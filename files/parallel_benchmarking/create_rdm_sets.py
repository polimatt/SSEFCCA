repeats = 20

if __name__ == '__main__':
    import pandas as pd

    #  Custom functions for SSEFCCA (Supervised Semi-Empirical Formula Compound Class Assignment)
    import sys
    sys.path.append('..')
    import ssefcca_functions as sfc

    #%%
    # Configuration settings and file paths

    # Output folder for comparison plots and CSV files
    data_folder = '../data'
    csv_folder = f'{data_folder}/csv_files'
    df_path = f'{csv_folder}/new_vk_areas_df.csv'
    df_with_pahs_path = f'{csv_folder}/new_vk_areas_with_pahs.csv'
    datasets_output_folder = f'{csv_folder}/benchmarking/train-test_datasets'

    # Model configuration flags
    weighted = True  # Whether to use class weights to handle imbalanced datasets
    stratify = True # Whether to stratify the train-test split

    # Get the df
    df = pd.read_csv(df_path)
    df_with_pahs = pd.read_csv(df_with_pahs_path)


    #%%

    # Define feature sets for different dimensionalities of van Krevelen space

    # 3D extended van Krevelen space (includes N/C)
    columns_multi = ['O/C', 'H/C', 'N/C', 'P/C', 'N/P']
    X = df[columns_multi].copy().to_numpy()

    # True compound class assignments
    y = df['category'].copy().to_numpy()

    # Create Rivas-Ubach categories (merge tannin and lignin into phytochemical)
    y_ru = df['category'].copy().to_numpy()
    y_ru[[x in ['tannin', 'lignin'] for x in y_ru]] = 'phytochemical'

    # now same but with PAHs
    columns_3d = ['O/C', 'H/C', 'N/C']
    X_with_pahs = df_with_pahs[columns_3d].copy().to_numpy()  # 3D features
    y_with_pahs = df_with_pahs['category'].copy().to_numpy()  # Target labels (including PAH)

    #%%

    def create_df(y,X,weights,X_cols):
        df = pd.DataFrame(y,columns=['y'])
        X_df = pd.DataFrame(X,columns=X_cols)
        df = pd.concat((df,X_df),axis=1)
        df['weights'] = weights
        return df


    #%%
    iteration_dict = {
        'non_r-u': {'X':X,'y':y},
        'r-u': {'X':X,'y':y_ru},
        'with_pahs': {'X':X_with_pahs,'y':y_with_pahs}
    }

    # create random slices of the df for training and testing; save them
    for i in range(1,repeats+1):
        for definition_type in iteration_dict:
            X_rdm_train, X_rdm_test, y_rdm_train, y_rdm_test, \
            weights_rdm_train, weights_rdm_test = sfc.train_test(iteration_dict[definition_type]['X'], iteration_dict[definition_type]['y'],
                                                                random_state=None, stratify_yn=stratify)

            combos = [[[y_rdm_train,X_rdm_train,weights_rdm_train],'train'],
                      [[y_rdm_test,X_rdm_test,weights_rdm_test],'test']]
            for combo in combos:
                sliced_df = create_df(*combo[0],X_cols=columns_3d if definition_type=='with_pahs' else columns_multi)
                sliced_df.to_csv(f'{datasets_output_folder}/{i}--{combo[1]}_{definition_type}.csv',index=False)