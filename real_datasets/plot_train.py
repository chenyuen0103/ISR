import pandas as pd
import matplotlib.pyplot as plt

# Load the training and validation data
# dataset = 'CelebA'
dataset = 'CUB'
model ='clip'
algo = 'HessianERM'
# algo = 'ERM'
seed = 0
train_df = pd.read_csv(f'../logs/{dataset}/{model}/{algo}/s{seed}/train.csv')
val_df = pd.read_csv(f'../logs/{dataset}/{model}/{algo}/s{seed}/val.csv')
test_df =  pd.read_csv(f'../logs/{dataset}/{model}/{algo}/s{seed}/test.csv')


# Calculate worst-case (minimum) accuracy per epoch for both training and validation data
worst_case_train_acc = train_df.groupby('epoch')['avg_acc'].min().reset_index()
worst_case_val_acc = val_df.groupby('epoch')['avg_acc'].min().reset_index()


# Plot Training and Validation Loss
plt.figure(figsize=(7, 5))
if 'Hessian' in algo:
    plt.plot(train_df['epoch'], train_df['hessian_aligned_loss'], label='Training')
    plt.plot(val_df['epoch'], val_df['hessian_aligned_loss'], label='Validation')
else:
    plt.plot(train_df['epoch'], train_df['avg_actual_loss'], label='Training')
    plt.plot(val_df['epoch'], val_df['avg_actual_loss'], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
if 'Hessian' in algo:
    plt.title(f'{dataset}--Hessian Aligned Loss')
else:
    plt.title(f'{dataset}--ERM Loss')
plt.legend()
# plt.xlim(0, 100)
# plt.ylim(0, 2)
# plt.savefig(f'../logs/{dataset}/{model}/{algo}/s{seed}/{dataset}_loss_scheduler.png')
plt.show()
plt.close()



# Function to find the worst group accuracy for each epoch
def get_worst_group_acc(df):
    # Extract only columns that contain group accuracy
    acc_columns = [col for col in df.columns if 'avg_acc_group' in col]
    # Find the minimum accuracy across these columns for each epoch
    worst_acc = df[acc_columns].min(axis=1)
    return df['epoch'], worst_acc

# Get worst-case accuracy for training and validation
train_epochs, train_worst_acc = get_worst_group_acc(train_df)
val_epochs, val_worst_acc = get_worst_group_acc(val_df)

# Plotting
plt.figure(figsize=(7, 5))
plt.plot(train_epochs, train_worst_acc, label='Training', linestyle='-')
plt.plot(val_epochs, val_worst_acc, label='Validation', linestyle='-')
plt.xlabel('Epoch')
plt.ylabel('Worst-case Group Accuracy')
plt.title(f'{dataset}--Worst-case Group Accuracy')
plt.legend()
# plt.xlim(0, 100)
# plt.savefig(f'../logs/{dataset}/{model}/{algo}/s{seed}/{dataset}_worst_group_acc_scheduler.png')
plt.show()
plt.close()

# Plot the difference between worst-case group accuracy between training and validation
plt.figure(figsize=(7, 5))
plt.plot(train_epochs, train_worst_acc - val_worst_acc, label='Training - Validation', linestyle='-')
plt.xlabel('Epoch')
plt.ylabel('Difference in Worst-case Group Accuracy')
plt.title(f'{dataset}--Difference in Worst-case Group Accuracy')
plt.legend()
# plt.xlim(0, 100)
# plt.savefig(f'../logs/{dataset}/{model}/{algo}/s{seed}/{dataset}_diff_worst_group_acc_scheduler.png')
plt.show()
plt.close()