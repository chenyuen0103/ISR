import pandas as pd
import glob
import os


def result_split(file_name = 'CUB_results_s1_combined.csv'):
    data_dir = './logs/ISR_hessian_results'


    df = pd.read_csv(os.path.join(data_dir, file_name))

    df = df.drop_duplicates()

    df_val = df[df['split'] == 'val']
    df_test = df[df['split'] == 'test']

    val_name = file_name.replace('.csv', '_val.csv')
    test_name = file_name.replace('.csv', '_test.csv')
    df_val.to_csv(os.path.join(data_dir, val_name), index=False)
    df_test.to_csv(os.path.join(data_dir, test_name), index=False)


def result_merge(file_name_start = 'CUB_results_s1'):
    data_dir = './logs/ISR_hessian_results'
    ISR_name = file_name_start + '_ISR.csv'
    hessian_name = file_name_start + '_hessian_exact.csv'
    df_ISR = pd.read_csv(os.path.join(data_dir, ISR_name))
    df_hessian = pd.read_csv(os.path.join(data_dir, hessian_name))

    df_combined = pd.merge(df_ISR, df_hessian, on=['dataset', 'algo', 'seed', 'ckpt', 'split', 'clf_type', 'C', 'pca_dim', 'd_spu', 'ISR_class', 'ISR_scale', 'env_label_ratio','num_iter'], how = 'outer', suffixes=('_ISR', '_hessian'))
    print(df_combined.head())
    df_combined.to_csv(os.path.join(data_dir, file_name_start + '_combined.csv'), index=False)


def merge_seeds(file_name_pattern='CUB_results_s*_hessian_exact.csv', data_dir='./logs/ISR_hessian_results'):
    # Create the full pattern for glob
    full_pattern = os.path.join(data_dir, file_name_pattern)

    # Use glob to get all file paths matching the pattern
    all_files = glob.glob(full_pattern)

    # Initialize an empty list to hold the dataframes
    df_list = []

    # Loop through the files, read each into a dataframe, and append it to the list
    for file_path in all_files:
        df = pd.read_csv(file_path)
        df_list.append(df)

    # Concatenate all the dataframes in the list
    merged_df = pd.concat(df_list, ignore_index=True)
    merged_df = merged_df[['dataset','seed','split','method','ISR_class','ISR_scale','num_iter', 'gradient_alpha',
                           'hessian_beta', 'acc-0', 'acc-1', 'acc-2', 'acc-3', 'worst_group', 'avg_acc', 'worst_acc']]

    grouped = merged_df.groupby(['dataset', 'split', 'method', 'ISR_class', 'ISR_scale',
                            'num_iter', 'gradient_alpha', 'hessian_beta']).agg({
        'avg_acc': ['mean', 'sem'],
        'worst_acc': ['mean', 'sem']
    })
    # Renaming the columns for clarity
    grouped.columns = ['avg_acc_mean', 'avg_acc_sem', 'worst_acc_mean', 'worst_acc_sem']
    # Resetting the index if you want the grouped columns back as regular columns
    grouped = grouped.reset_index()
    cleaned_grouped = grouped.dropna(subset=['avg_acc_sem', 'worst_acc_sem'])
    # Display the cleaned DataFrame
    print(cleaned_grouped)
    val = cleaned_grouped[cleaned_grouped['split'] == 'val']
    test = cleaned_grouped[cleaned_grouped['split'] == 'test']
    num_runs = len(all_files)
    dataset = file_name_pattern.split('_')[0]
    val.to_csv(f'{data_dir}/{dataset}_{num_runs}runs_val.csv', index=False)
    test.to_csv(f'{data_dir}/{dataset}_{num_runs}runs_test.csv', index=False)
    print(f"Saved {dataset}_{num_runs}runs_val.csv and {dataset}_{num_runs}runs_test.csv in {data_dir}")
    return val, test

def find_best_isr(data_dir='./logs/ISR_hessian_results', file_name='CUB_5runs_val.csv', worst_case = False):
    val = pd.read_csv(os.path.join(data_dir, file_name))
    test = pd.read_csv(os.path.join(data_dir, file_name.replace('val', 'test')))
    val = val[val['gradient_alpha'] == 0]
    test = test[test['gradient_alpha'] == 0]
    val = val[val['hessian_beta'] == 0]
    test = test[test['hessian_beta'] == 0]
    best_hyperparameters, best_test_performance = find_best_hps(val, test, worst_case)

    return best_hyperparameters, best_test_performance

def find_best_gm(data_dir='./logs/ISR_hessian_results', file_name='CUB_5runs_val.csv', worst_case = False):
    val = pd.read_csv(os.path.join(data_dir, file_name))
    test = pd.read_csv(os.path.join(data_dir, file_name.replace('val', 'test')))
    val = val[val['hessian_beta'] == 0]
    test = test[test['hessian_beta'] == 0]
    val = val[val['gradient_alpha'] != 0]
    test = test[test['gradient_alpha'] != 0]
    best_hyperparameters, best_test_performance = find_best_hps(val, test, worst_case)

    return best_hyperparameters, best_test_performance

def find_best_hm(data_dir='./logs/ISR_hessian_results', file_name='CUB_5runs_val.csv', worst_case = False):
    val = pd.read_csv(os.path.join(data_dir, file_name))
    test = pd.read_csv(os.path.join(data_dir, file_name.replace('val', 'test')))
    val = val[val['gradient_alpha'] == 0]
    test = test[test['gradient_alpha'] == 0]
    val = val[val['hessian_beta'] != 0]
    test = test[test['hessian_beta'] != 0]
    best_hyperparameters, best_test_performance = find_best_hps(val, test, worst_case)

    return best_hyperparameters, best_test_performance

def find_best_gm_hm(data_dir='./logs/ISR_hessian_results', file_name='CUB_5runs_val.csv', worst_case = False):
    val = pd.read_csv(os.path.join(data_dir, file_name))
    test = pd.read_csv(os.path.join(data_dir, file_name.replace('val', 'test')))
    val = val[val['gradient_alpha'] != 0]
    test = test[test['gradient_alpha'] != 0]
    val = val[val['hessian_beta'] != 0]
    test = test[test['hessian_beta'] != 0]
    best_hyperparameters, best_test_performance = find_best_hps(val, test, worst_case)

    return best_hyperparameters, best_test_performance

def find_best_hps(val_df, test_df, worst_case = False):
    col_mean = 'worst_acc_mean' if worst_case else 'avg_acc_mean'
    col_sem = 'worst_acc_sem' if worst_case else 'avg_acc_sem'
    # Step 1: Identify the highest average accuracy
    max_avg_acc_mean = val_df[col_mean].max()

    # Filter to include only those with the highest average accuracy
    top_performers = val_df[val_df[col_mean] == max_avg_acc_mean]

    # Step 2: Select based on Standard Error
    # Identify the entry with the smallest standard error among the top performers
    best_val_hyperparameters = top_performers.loc[top_performers[col_sem].idxmin()]

    # Display the chosen hyperparameters

    print("Chosen hyperparameters for best",f"{'worst case accuracy' if worst_case else 'average accuracy'}"  ":\n", best_val_hyperparameters)

    # Extracting hyperparameters
    best_hyperparameters = {
        'ISR_class': best_val_hyperparameters['ISR_class'],
        'ISR_scale': best_val_hyperparameters['ISR_scale'],
        'num_iter': best_val_hyperparameters['num_iter'],
        'gradient_alpha': best_val_hyperparameters['gradient_alpha'],
        'hessian_beta': best_val_hyperparameters['hessian_beta']
    }

    # Filter the test set for these hyperparameters
    best_test_performance = test_df[
        (test_df['ISR_class'] == best_hyperparameters['ISR_class']) &
        (test_df['ISR_scale'] == best_hyperparameters['ISR_scale']) &
        (test_df['num_iter'] == best_hyperparameters['num_iter']) &
        (test_df['gradient_alpha'] == best_hyperparameters['gradient_alpha']) &
        (test_df['hessian_beta'] == best_hyperparameters['hessian_beta'])
    ]
    pd.set_option('display.max_columns', None)

    # Display the performance on the test set
    print("Performance on test for best",f"{'worst case accuracy' if worst_case else 'average accuracy'}"  ":\n", best_test_performance[['dataset', 'split','gradient_alpha','hessian_beta','avg_acc_mean', 'avg_acc_sem', 'worst_acc_mean', 'worst_acc_sem']])

    return best_hyperparameters, best_test_performance




def main():
    cub_pattern = 'CUB_results_s*_hessian_exact.csv'
    celeba_pattern = 'CelebA_results_s*_hessian_exact.csv'

    cubs_val, cubs_test = merge_seeds()
    celeba_val, celeba_test = merge_seeds(file_name_pattern=celeba_pattern)
    # print(cubs_val)
    find_best_isr(worst_case=True, file_name='CelebA_5runs_val.csv')
    find_best_gm(worst_case=True, file_name='CelebA_5runs_val.csv')
    find_best_hm(worst_case=True, file_name='CelebA_5runs_val.csv')
    find_best_gm_hm(worst_case=True, file_name='CelebA_5runs_val.csv')

if __name__ == '__main__':
    main()