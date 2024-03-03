import pandas as pd
import os
from itertools import product

# Define the parameters to iterate over
# datasets = ['CUB']  # Add more if needed
datasets = ['CUB']  # Example datasets
models = ['clip_512', 'clip', 'vits', 'resnet50']  # Example models
# models = ['vits']  # Example models
algos = ['HessianERM', 'ERM']  # Example algorithms
seeds = range(22)  # Example seed values
grad_alphas = [10**-i for i in range(10)] + [0]  # 1, 1e-1, ..., 1e-9, 0
hess_betas = [10**-i for i in range(10)] + [0]
scheduler_options = [True, False]  # Scheduler options


# Function to find the worst group accuracy for each epoch
def get_worst_group_acc(df):
    acc_columns = [col for col in df.columns if 'avg_acc_group' in col]
    worst_acc = df[acc_columns].min(axis=1)
    return worst_acc.mean()


# Placeholder for the results
results = []

# Generate combinations using itertools.product
combinations = product(datasets, models, algos, seeds, grad_alphas, hess_betas, scheduler_options)

for dataset, model, algo, seed, grad_alpha, hess_beta, scheduler in combinations:
    grad_alpha_formatted = "{:.1e}".format(grad_alpha).replace('.0e', 'e')
    hess_beta_formatted = "{:.1e}".format(hess_beta).replace('.0e', 'e')
    scheduler_suffix = '' if scheduler else '_no_scheduler'

    # Construct the directory path
    dir_path = f"./logs/{dataset}/{model}/{algo}/s{seed}/grad_alpha_{grad_alpha_formatted}_hess_beta_{hess_beta_formatted}{scheduler_suffix}"

    # Check if the directory exists
    if os.path.exists(dir_path):
        test_csv_path = os.path.join(dir_path, 'test.csv')

        # Load the test.csv if it exists and has rows
        if os.path.isfile(test_csv_path) and os.path.getsize(test_csv_path) > 0:
            test_df = pd.read_csv(test_csv_path)
            # Check if the DataFrame is empty
            if not test_df.empty:
                avg_test_acc = test_df['avg_acc'].mean()
                avg_worst_case_acc = get_worst_group_acc(test_df)
                num_epochs_trained = test_df['epoch'].max()  # Extract the maximum epoch

                # Append the results
                results.append({
                    'dataset': dataset,
                    'model': model,
                    'algo': algo,
                    'seed': seed,
                    'grad_alpha': grad_alpha,
                    'hess_beta': hess_beta,
                    'scheduler': scheduler,
                    'average_test_accuracy': avg_test_acc,
                    'average_worst_case_accuracy': avg_worst_case_acc,
                    'num_epochs_trained': num_epochs_trained  # Add the number of epochs trained
                })

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Save to CSV
results_df.to_csv('model_performance_summary_new_hess.csv', index=False)

print("CSV file created successfully.")