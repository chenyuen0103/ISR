import pandas as pd
import matplotlib.pyplot as plt

# Load the training and validation data
# dataset = 'CelebA'
dataset = 'CUB'
model ='clip'
algo = 'HessianERM'
seed = 2
train_df = pd.read_csv(f'../logs/{dataset}/{model}/{algo}/s{seed}/train.csv')
val_df = pd.read_csv(f'../logs/{dataset}/{model}/{algo}/s{seed}/val.csv')
test_df =  pd.read_csv(f'../logs/{dataset}/{model}/{algo}/s{seed}/test.csv')


# Calculate worst-case (minimum) accuracy per epoch for both training and validation data
worst_case_train_acc = train_df.groupby('epoch')['avg_acc'].min().reset_index()
worst_case_val_acc = val_df.groupby('epoch')['avg_acc'].min().reset_index()


# Plot Training and Validation Loss
plt.figure(figsize=(7, 5))
plt.plot(train_df['epoch'], train_df['hessian_aligned_loss'], label='Training')
plt.plot(val_df['epoch'], val_df['hessian_aligned_loss'], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Hessian Aligned Loss')
plt.legend()
# plt.xlim(0, 300)
plt.savefig(f'../logs/{dataset}/{model}/{algo}/s{seed}/{dataset}_loss.pdf')
plt.show()
plt.close()


# Plot Worst-case Training and Validation Accuracy
plt.figure(figsize=(7, 5))
plt.plot(worst_case_train_acc['epoch'], worst_case_train_acc['avg_acc'], label='Training', linestyle='-')
plt.plot(worst_case_val_acc['epoch'], worst_case_val_acc['avg_acc'], label='Validation', linestyle='-')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Worst-case Accuracy')
plt.legend()
# plt.xlim(0, 300)
plt.savefig(f'../logs/{dataset}/{model}/{algo}/s{seed}/{dataset}_acc.pdf')
plt.show()
plt.close()