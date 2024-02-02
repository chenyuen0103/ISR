import os
from itertools import product
from tqdm import tqdm
from configs import get_train_command
import pdb
# gpu_idx = 0,1,2,3  # could be None if you want to use cpu

# algos = ['ERM','reweight','groupDRO']
algos = ['ERM']
# dataset = 'MultiNLI'  # could be 'CUB' (i.e., Waterbirds), 'CelebA' or 'MultiNLI'
dataset = 'CelebA'
# can add some suffix to the algo name to flag the version,
# e.g., with algo_suffix = "-my_version", the algo name becomes "ERM-my_version"
algo_suffix = ""
# Assuming seeds, algos, dataset, and get_train_command are defined

gpu_count = 4  # Number of GPUs available
gpu_idx = 0   # Start with GPU 0
if dataset == 'CelebA':
    gpu_idx = 1
# seeds = range(10)
seeds = [0]
for seed, algo in tqdm(list(product(seeds, algos)), desc='Experiments'):
    # Generate the command
    command = get_train_command(dataset=dataset, algo=algo, gpu_idx=gpu_idx, seed=seed,
                                save_best=True, save_last=True)
    print('Command:', command)

    # Run the command in the background
    os.system(command)

    # Rotate the GPU index
    # gpu_idx = (gpu_idx + 1) % gpu_count

    # Optional: Introduce a slight delay if needed
    # time.sleep(1)



# for seed, algo in tqdm(list(product(seeds, algos)), desc='Experiments'):
#     command = get_train_command(dataset=dataset, algo=algo, gpu_idx=gpu_idx, seed=seed,
#                       save_best=True, save_last=True)
#     print('Command:', command)
#     os.system(command)
