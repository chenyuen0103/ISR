import argparse
import os
import pickle
from itertools import product

import numpy as np
import torch
import torch.nn as nn
import torchvision
from tqdm.auto import tqdm
import configs
from configs.model_config import model_attributes
from data import dataset_attributes, shift_types, prepare_data, log_data
from utils.train_utils import check_args
from utils.train_utils import set_seed
import open_clip
import pdb
from torch.optim import AdamW
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from utils.train_utils import set_seed, Logger, CSVBatchLogger, log_args
from utils.loss_utils import LossComputer
import pandas as pd

import pickle
from itertools import product
import gc

parser = argparse.ArgumentParser()



# Confounders
parser.add_argument('-t', '--target_name', default="waterbird_complete95")
parser.add_argument('-c', '--confounder_names', nargs='+', default=['forest2water2'])

# Resume?
parser.add_argument('--resume', default=False, action='store_true')

# Label shifts
parser.add_argument('--minority_fraction', type=float)
parser.add_argument('--imbalance_ratio', type=float)

# Data

parser.add_argument('-d', '--dataset',
                    choices=dataset_attributes.keys(),
                    default='CUB')
parser.add_argument('-s', '--shift_type',
                    choices=shift_types,
                    default='confounder')
parser.add_argument(
    '--model',
    choices=model_attributes.keys(),
    default='resnet50')
parser.add_argument('--fraction', type=float, default=1.0)
parser.add_argument('--root_dir', default=None)
parser.add_argument('--reweight_groups', action='store_true', default=False)
parser.add_argument('--augment_data', action='store_true', default=False)
parser.add_argument('--val_fraction', type=float, default=0.1)
parser.add_argument('--algo', default='Hessian')
parser.add_argument('--train_from_scratch', action='store_true', default=False)
parser.add_argument('--alpha', default=1e-5, type=float, help='gradient hyperparameter')
parser.add_argument('--beta', default=1e-2, type=float, help='hessian hyperparameter')
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--log_dir', default=configs.LOG_FOLDER)
parser.add_argument('--algo_suffix', type=str, default='CLIP', help='The suffix of log folder name')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--btl', default=False, action='store_true')
parser.add_argument('--hinge', default=False, action='store_true')
args = parser.parse_args()
# check_args(args)

model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K')
tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K')

# args.log_dir = os.path.join(args.log_dir, args.dataset, args.algo, f's{args.seed}')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# set_seed(args.seed)

test_data = None
test_loader = None
if args.shift_type == 'confounder':
    train_data, val_data, test_data = prepare_data(args, train=True)
elif args.shift_type == 'label_shift_step':
    train_data, val_data = prepare_data(args, train=True)

loader_kwargs = {'batch_size': args.batch_size,
                 'num_workers': 4, 'pin_memory': True}
train_loader = train_data.get_loader(
    train=True, reweight_groups=args.reweight_groups, **loader_kwargs)
val_loader = val_data.get_loader(
    train=False, reweight_groups=None, **loader_kwargs)
if test_data is not None:
    test_loader = test_data.get_loader(
        train=False, reweight_groups=None, **loader_kwargs)

data = {}
data['train_loader'] = train_loader
data['val_loader'] = val_loader
data['test_loader'] = test_loader
data['train_data'] = train_data
data['val_data'] = val_data
data['test_data'] = test_data
n_classes = train_data.n_classes

if args.algo == 'ERM':
    criterion = nn.CrossEntropyLoss(reduction='none')
elif args.algo == 'Hessian':
    criterion = nn.CrossEntropyLoss(reduction='none')

use_clip = type(model).__name__ == 'CLIP'
save_prefix = 'CLIP_' if use_clip else ''
load_ckpt = not use_clip
#%%
load_ckpt = False
def run_epoch(epoch, model, optimizer, loader, loss_computer, logger, csv_logger, args,
              is_training, show_progress=False, log_every=50, scheduler=None):
    """
    scheduler is only used inside this function if model is bert.
    """

    if is_training:
        print("Start Training")
        breakpoint()
        model.train()
        if args.model == 'bert':
            model.zero_grad()
        print("End training")
    else:
        model.eval()

    if show_progress:
        prog_bar_loader = tqdm(loader)
    else:
        prog_bar_loader = loader

    with torch.set_grad_enabled(is_training):
        for batch_idx, batch in enumerate(prog_bar_loader):
            batch = tuple(t.to(device) for t in batch)
            # batch = tuple(t.cuda() for t in batch)
            x = batch[0]
            y = batch[1]
            g = batch[2]
            if args.model == 'bert':
                input_ids = x[:, :, 0]
                input_masks = x[:, :, 1]
                segment_ids = x[:, :, 2]
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=input_masks,
                    token_type_ids=segment_ids,
                    labels=y
                )[1]  # [1] returns logits
            else:
                outputs = model(x)

            if args.algo == 'Hessian':
                loss_main, erm_loss, hess_loss, grad_loss = loss_computer.exact_hessian_loss(model,x , y, g)
            else:
                loss_main = loss_computer.loss(outputs, y, g, is_training)

            if is_training:
                if args.model == 'bert':
                    loss_main.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                else:
                    optimizer.zero_grad()
                    loss_main.backward()
                    optimizer.step()

            if is_training and (batch_idx + 1) % log_every == 0:
                csv_logger.log(epoch, batch_idx, loss_computer.get_stats(model, args))
                csv_logger.flush()
                loss_computer.log_stats(logger, is_training)
                loss_computer.reset_stats()

        if (not is_training) or loss_computer.batch_count > 0:
            csv_logger.log(epoch, batch_idx, loss_computer.get_stats(model, args))
            csv_logger.flush()
            loss_computer.log_stats(logger, is_training)
            if is_training:
                loss_computer.reset_stats()




def train(model, criterion, dataset,
          logger, train_csv_logger, val_csv_logger, test_csv_logger,
          args, epoch_offset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)
    breakpoint()
    model = model.cuda()

    # process generalization adjustment stuff
    adjustments = [float(c) for c in args.generalization_adjustment.split(',')]
    assert len(adjustments) in (1, dataset['train_data'].n_groups)
    if len(adjustments) == 1:
        adjustments = np.array(adjustments * dataset['train_data'].n_groups)
    else:
        adjustments = np.array(adjustments)

    train_loss_computer = LossComputer(
        criterion,
        is_robust=args.robust,
        dataset=dataset['train_data'],
        alpha=args.alpha,
        gamma=args.gamma,
        adj=adjustments,
        step_size=args.robust_step_size,
        normalize_loss=args.use_normalized_loss,
        btl=args.btl,
        min_var_weight=args.minimum_variational_weight)

    # BERT uses its own scheduler and optimizer
    if args.model == 'bert':
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=args.lr,
            eps=args.adam_epsilon)
        t_total = len(dataset['train_loader']) * args.n_epochs
        print(f'\nt_total is {t_total}\n')
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=t_total)
    else:
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay)
        if args.scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                'min',
                factor=0.1,
                patience=5,
                threshold=0.0001,
                min_lr=0,
                eps=1e-08)
        else:
            scheduler = None

    best_val_acc = 0
    best_avg_val_acc = 0
    for epoch in range(epoch_offset, epoch_offset + args.n_epochs):
        logger.write('\nEpoch [%d]:\n' % epoch)
        logger.write(f'Training:\n')
        run_epoch(
            epoch, model, optimizer,
            dataset['train_loader'],
            train_loss_computer,
            logger, train_csv_logger, args,
            is_training=True,
            show_progress=args.show_progress,
            log_every=args.log_every,
            scheduler=scheduler)

        logger.write(f'\nValidation:\n')
        val_loss_computer = LossComputer(
            criterion,
            is_robust=args.robust,
            dataset=dataset['val_data'],
            step_size=args.robust_step_size,
            alpha=args.alpha)
        run_epoch(
            epoch, model, optimizer,
            dataset['val_loader'],
            val_loss_computer,
            logger, val_csv_logger, args,
            is_training=False)

        # Test set; don't print to avoid peeking
        if dataset['test_data'] is not None:
            test_loss_computer = LossComputer(
                criterion,
                is_robust=args.robust,
                dataset=dataset['test_data'],
                step_size=args.robust_step_size,
                alpha=args.alpha)
            run_epoch(
                epoch, model, optimizer,
                dataset['test_loader'],
                test_loss_computer,
                None, test_csv_logger, args,
                is_training=False)

        # Inspect learning rates
        if (epoch + 1) % 1 == 0:
            for param_group in optimizer.param_groups:
                curr_lr = param_group['lr']
                logger.write('Current lr: %f\n' % curr_lr)

        if args.scheduler and args.model != 'bert':
            if args.robust:
                val_loss, _ = val_loss_computer.compute_robust_loss_greedy(val_loss_computer.avg_group_loss,
                                                                           val_loss_computer.avg_group_loss)
            else:
                val_loss = val_loss_computer.avg_actual_loss
            scheduler.step(val_loss)  # scheduler step to update lr at the end of epoch

        # if epoch % args.save_step == 0:
        #     torch.save(model, os.path.join(args.log_dir, '%d_model.pth' % epoch))

        if args.save_last:
            torch.save(model, os.path.join(args.log_dir, 'last_model.pth'))

        if args.save_best:

            curr_val_acc = min(val_loss_computer.avg_group_acc)
            curr_avg_val_acc = val_loss_computer.avg_acc

            logger.write(f'Current average validation accuracy: {curr_avg_val_acc}\n')
            logger.write(f'Current worst-group validation accuracy: {curr_val_acc}\n')
            if curr_val_acc > best_val_acc:
                best_val_acc = curr_val_acc
                torch.save(model, os.path.join(args.log_dir, 'best_model.pth'))
                logger.write(f'Best worst-group model saved at epoch {epoch}\n')
            if curr_avg_val_acc > best_avg_val_acc:
                best_avg_val_acc = curr_avg_val_acc
                torch.save(model, os.path.join(args.log_dir, 'best_avg_acc_model.pth'))
                logger.write(f'Best average-accuracy model saved at epoch {epoch}\n')

        if args.automatic_adjustment:
            gen_gap = val_loss_computer.avg_group_loss - train_loss_computer.exp_avg_loss
            adjustments = gen_gap * torch.sqrt(train_loss_computer.group_counts)
            train_loss_computer.adj = adjustments
            logger.write('Adjustments updated\n')
            for group_idx in range(train_loss_computer.n_groups):
                logger.write(
                    f'  {train_loss_computer.get_group_name(group_idx)}:\t'
                    f'adj = {train_loss_computer.adj[group_idx]:.3f}\n')
        logger.write('\n')



args.log_dir = os.path.join(args.log_dir, args.dataset, args.algo + args.algo_suffix, f's{args.seed}')

if os.path.exists(args.log_dir) and args.resume:
    resume = True
    mode = 'a'
else:
    resume = False
    mode = 'w'


## Initialize logs
if not os.path.exists(args.log_dir):
    print(args.log_dir)
    os.makedirs(args.log_dir)

logger = Logger(os.path.join(args.log_dir, 'log.txt'), mode)
# Record args
log_args(args, logger)

set_seed(args.seed)


log_data(data, logger)

## Initialize model
pretrained = not args.train_from_scratch
if resume:
    model = torch.load(os.path.join(args.log_dir, 'last_model.pth'))
    d = train_data.input_size()[0]
elif model_attributes[args.model]['feature_type'] in ('precomputed', 'raw_flattened'):
    assert pretrained
    # Load precomputed features
    d = train_data.input_size()[0]
    model = nn.Linear(d, n_classes)
    model.has_aux_logits = False
elif args.model == 'resnet50':
    model = torchvision.models.resnet50(pretrained=pretrained)
    d = model.fc.in_features
    model.fc = nn.Linear(d, n_classes)
elif args.model == 'resnet34':
    model = torchvision.models.resnet34(pretrained=pretrained)
    d = model.fc.in_features
    model.fc = nn.Linear(d, n_classes)
elif args.model == 'wideresnet50':
    model = torchvision.models.wide_resnet50_2(pretrained=pretrained)
    d = model.fc.in_features
    model.fc = nn.Linear(d, n_classes)
elif args.model == 'bert':
    assert args.dataset == 'MultiNLI'

    from transformers import BertConfig, BertForSequenceClassification

    config_class = BertConfig
    model_class = BertForSequenceClassification

    config = config_class.from_pretrained(
        'bert-base-uncased',
        num_labels=3,
        finetuning_task='mnli')
    model = model_class.from_pretrained(
        'bert-base-uncased',
        from_tf=False,
        config=config)
else:
    raise ValueError('Model not recognized.')

logger.flush()

## Define the objective
if args.hinge:
    assert args.dataset in ['CelebA', 'CUB']  # Only supports binary


    def hinge_loss(yhat, y):
        # The torch loss takes in three arguments so we need to split yhat
        # It also expects classes in {+1.0, -1.0} whereas by default we give them in {0, 1}
        # Furthermore, if y = 1 it expects the first input to be higher instead of the second,
        # so we need to swap yhat[:, 0] and yhat[:, 1]...
        torch_loss = torch.nn.MarginRankingLoss(margin=1.0, reduction='none')
        y = (y.float() * 2.0) - 1.0
        return torch_loss(yhat[:, 1], yhat[:, 0], y)


    criterion = hinge_loss
else:
    criterion = torch.nn.CrossEntropyLoss(reduction='none')



if resume:
    df = pd.read_csv(os.path.join(args.log_dir, 'test.csv'))
    epoch_offset = df.loc[len(df) - 1, 'epoch'] + 1
    logger.write(f'starting from epoch {epoch_offset}')
else:
    epoch_offset = 0
train_csv_logger = CSVBatchLogger(os.path.join(args.log_dir, 'train.csv'), train_data.n_groups, mode=mode)
val_csv_logger = CSVBatchLogger(os.path.join(args.log_dir, 'val.csv'), train_data.n_groups, mode=mode)
test_csv_logger = CSVBatchLogger(os.path.join(args.log_dir, 'test.csv'), train_data.n_groups, mode=mode)

train(model, criterion, data, logger, train_csv_logger, val_csv_logger, test_csv_logger, args,
      epoch_offset=epoch_offset)