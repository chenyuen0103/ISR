import pandas as pd
import matplotlib.pyplot as plt

# Load the training and validation data
dataset = 'CUB'
model ='clip'
algo = 'HessianERM'
seed = 0
train_df = pd.read_csv(f'../logs/{dataset}/{model}/{algo}/s{seed}/train.csv')
val_df = pd.read_csv(f'../logs/{dataset}/{model}/{algo}/s{seed}/val.csv')
test_df =  pd.read_csv(f'../logs/{dataset}/{model}/{algo}/s{seed}/test.csv')


# Calculate worst-case (minimum) accuracy per epoch for both training and validation data
worst_case_train_acc = train_df.groupby('epoch')['avg_acc'].min().reset_index()
worst_case_val_acc = val_df.groupby('epoch')['avg_acc'].min().reset_index()

plt.figure(figsize=(14, 7))

# Plot for Training and Validation Loss
plt.subplot(1, 2, 1)
plt.plot(train_df['epoch'], train_df['avg_actual_loss'], label='Training Loss')
plt.plot(val_df['epoch'], val_df['avg_actual_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.xlim(0, 300)
plt.savefig(f'../logs/{dataset}/{model}/{algo}/s{seed}/train_val_loss.pdf')  # Save the figure
plt.close()

# Plot for Worst-case Training and Validation Accuracy
plt.subplot(1, 2, 2)
plt.plot(worst_case_train_acc['epoch'], worst_case_train_acc['avg_acc'], label='Worst-case Training Accuracy', linestyle='-')
plt.plot(worst_case_val_acc['epoch'], worst_case_val_acc['avg_acc'], label='Worst-case Validation Accuracy', linestyle='-')
plt.xlabel('Epoch')
plt.ylabel('Worst-case Accuracy')
plt.title('Worst-case Training and Validation Accuracy')
plt.legend()
plt.xlim(0, 300)
plt.savefig(f'../logs/{dataset}/{model}/{algo}/s{seed}/train_val_acc.pdf')  # Save the figure
plt.close()
plt.tight_layout()
plt.show()
