import os
import pandas as pd
import glob


patterns = ['CelebA_results_s*_hessian_exact.csv','CUB_results_s*_hessian_exact.csv','MultiNLI_results_s*_hessian_exact.csv']
data_dir = './logs/ISR_Hessian_results_new'
print(os.getcwd())
for pattern in patterns:
    files = glob.glob(os.path.join(data_dir, pattern))
    for file in files:
        df = pd.read_csv(file)
        df = df[df['penalty_anneal_iters'] == 0]
        df.to_csv(file, index=False)
