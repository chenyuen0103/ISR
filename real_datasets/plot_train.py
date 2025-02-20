import pandas as pd
import matplotlib.pyplot as plt

# Load the training and validation data
# dataset = 'CelebA'
dataset = 'CUB'
# model ='clip'
# model ='clip512'
model = 'vits'
# model = 'resnet50'
# algo = 'HessianERM'
algo = 'ERM'
seed = 0
scheduler = True
lr = 3e-2
# lr = None
batch_size = 512
# batch_size = None
# scheduler = False
grad_alpha = 1e-4
# grad_alpha = 0
# hess_beta = 0
hess_beta = 1e-4

grad_alpha_formatted = "{:.1e}".format(grad_alpha).replace('.0e', 'e')
hess_beta_formatted = "{:.1e}".format(hess_beta).replace('.0e', 'e')

if lr is not None and batch_size is None:
    lr_formatted = "{:.1e}".format(lr).replace('.0e', 'e')
    train_df = pd.read_csv(
        f"../logs/{dataset}/{model}/{algo}/lr{lr_formatted}/s{seed}/grad_alpha_{grad_alpha_formatted}_hess_beta_{hess_beta_formatted}{'_no_scheduler' if not scheduler else ''}/train.csv")
    val_df = pd.read_csv(
        f"../logs/{dataset}/{model}/{algo}/lr{lr_formatted}/s{seed}/grad_alpha_{grad_alpha_formatted}_hess_beta_{hess_beta_formatted}{'_no_scheduler' if not scheduler else ''}/val.csv")
    test_df = pd.read_csv(
        f"../logs/{dataset}/{model}/{algo}/lr{lr_formatted}/s{seed}/grad_alpha_{grad_alpha_formatted}_hess_beta_{hess_beta_formatted}{'_no_scheduler' if not scheduler else ''}/test.csv")
elif lr is not None:
    lr_formatted = "{:.1e}".format(lr).replace('.0e', 'e')
    train_df = pd.read_csv(
        f"../logs/{dataset}/{model}/{algo}/lr{lr_formatted}_batchsize_{batch_size}/s{seed}/grad_alpha_{grad_alpha_formatted}_hess_beta_{hess_beta_formatted}{'_no_scheduler' if not scheduler else ''}/train.csv")
    val_df = pd.read_csv(
        f"../logs/{dataset}/{model}/{algo}/lr{lr_formatted}_batchsize_{batch_size}/s{seed}/grad_alpha_{grad_alpha_formatted}_hess_beta_{hess_beta_formatted}{'_no_scheduler' if not scheduler else ''}/val.csv")
    test_df = pd.read_csv(
        f"../logs/{dataset}/{model}/{algo}/lr{lr_formatted}_batchsize_{batch_size}/s{seed}/grad_alpha_{grad_alpha_formatted}_hess_beta_{hess_beta_formatted}{'_no_scheduler' if not scheduler else ''}/test.csv")
else:
    train_df = pd.read_csv(f"../logs/{dataset}/{model}/{algo}/s{seed}/grad_alpha_{grad_alpha_formatted}_hess_beta_{hess_beta_formatted}{'_no_scheduler' if not scheduler else ''}/train.csv")
    val_df = pd.read_csv(f"../logs/{dataset}/{model}/{algo}/s{seed}/grad_alpha_{grad_alpha_formatted}_hess_beta_{hess_beta_formatted}{'_no_scheduler' if not scheduler else ''}/val.csv")
    test_df =  pd.read_csv(f"../logs/{dataset}/{model}/{algo}/s{seed}/grad_alpha_{grad_alpha_formatted}_hess_beta_{hess_beta_formatted}{'_no_scheduler' if not scheduler else ''}/test.csv")
# train_df = pd.read_csv(f'../logs/{dataset}/{model}/{algo}/s{seed}/train.csv')
# val_df = pd.read_csv(f'../logs/{dataset}/{model}/{algo}/s{seed}/val.csv')
# test_df =  pd.read_csv(f'../logs/{dataset}/{model}/{algo}/s{seed}/test.csv')

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
# plt.xlim(0, 20)
# plt.ylim(0, 2)
# plt.savefig(f'../logs/{dataset}/{model}/{algo}/s{seed}/{dataset}_loss_scheduler.png')
plt.show()
plt.close()




# plot average accuracy of training and validation
plt.figure(figsize=(7, 5))
plt.plot(train_df['epoch'], train_df['avg_acc'], label='Training', linestyle='-')
plt.plot(val_df['epoch'], val_df['avg_acc'], label='Validation', linestyle='-')
plt.xlabel('Epoch')
plt.ylabel('Average Group Accuracy')
plt.title(f'{dataset}--Average Group Accuracy')
plt.legend()
# plt.xlim(0, 20)
# plt.savefig(f'../logs/{dataset}/{model}/{algo}/s{seed}/{dataset}_avg_group_acc_scheduler.png')
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
# plt.xlim(0, 20)
# plt.savefig(f'../logs/{dataset}/{model}/{algo}/s{seed}/{dataset}_worst_group_acc_scheduler.png')
plt.show()
plt.close()





# print the result arguments
print(f'grad_alpha: {grad_alpha}')
print(f'hess_beta: {hess_beta}')
print(f'scheduler: {scheduler}')
print(f'seed: {seed}')
print(f'algo: {algo}')
print(f'model: {model}')
print(f'dataset: {dataset}')
# print average and worst-case and Test Accuracy at the end of training
print(f'Average Test Accuracy: {test_df["avg_acc"].iloc[-1]}')
_, test_worst = get_worst_group_acc(test_df)
print(f'Worst-case Test Accuracy: {test_worst.iloc[-1]}')
