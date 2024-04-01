import pandas as pd
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


def result_merge():
    data_dir = './logs/ISR_hessian_results'
    file_name_start = 'CUB_results_s1'
    ISR_name = file_name_start + '_ISR.csv'
    hessian_name = file_name_start + '_hessian_exact.csv'
    df_ISR = pd.read_csv(os.path.join(data_dir, ISR_name))
    df_hessian = pd.read_csv(os.path.join(data_dir, hessian_name))

    df_combined = pd.merge(df_ISR, df_hessian, on=['dataset', 'algo', 'seed', 'ckpt', 'split', 'clf_type', 'C', 'pca_dim', 'd_spu', 'ISR_class', 'ISR_scale', 'env_label_ratio','num_iter'], how = 'outer', suffixes=('_ISR', '_hessian'))
    print(df_combined.head())
    df_combined.to_csv(os.path.join(data_dir, file_name_start + '_combined.csv'), index=False)


def main():
    result_split()
    # result_merge()

if __name__ == '__main__':
    main()