import os
from itertools import chain

import numpy as np
import torch
import timm
# Set the default CUDA device to GPU 2
# torch.cuda.set_device(2)
from torch.optim import AdamW
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score
from utils.loss_utils import LossComputer
from utils.train_utils import set_seed
import pdb


# torch.autograd.set_detect_anomaly(True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
max_process_batch = 32


class LogisticRegression(torch.nn.Module):
    # build the constructor
    def __init__(self, n_inputs, n_outputs):
        super().__init__()
        self.linear = torch.nn.Linear(n_inputs, n_outputs, bias = False)

    # make predictions
    def forward(self, x):
        # Just return the logits (raw scores). Softmax will be applied in the loss function.
        # if not isinstance(x, torch.Tensor):
        #     x = torch.tensor(x).float()
        if x.dim() == 1:
            x = x.view(1, -1)
        # print(x.shape)
        return self.linear(x)

    def predict(self, x):
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)


    def score(self, X, y, sample_weight=None):
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)


def run_epoch(epoch, model, clf, optimizer, loader, loss_computer, logger, csv_logger, args,
              is_training, show_progress=False, log_every=50, scheduler=None):
    """
    scheduler is only used inside this function if model is bert.
    """
    if is_training:
        print("Start Training")
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
            optimizer.zero_grad()
            # batch = tuple(t.cuda() for t in batch)
            x_batch = batch[0]
            y_batch = batch[1]
            g_batch = batch[2]
            # x = batch[0]
            # y = batch[1]
            # g = batch[2]
            num_sub_batches = len(x_batch) // max_process_batch
            for sub_batch_idx in range(num_sub_batches):
                start_idx = sub_batch_idx * max_process_batch
                end_idx = start_idx + max_process_batch
                x, y, g = x_batch[start_idx:end_idx], y_batch[start_idx:end_idx], g_batch[start_idx:end_idx]

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
                    # breakpoint()
                elif args.model == 'vits':
                    # Reshape x to [batch_size, channels, height, width]
                    # encoder = model.encoder
                    x = x.view(x.size(0), 3, 224, 224)
                    outputs = model(x)
                    logits = clf(outputs)
                    # breakpoint()
                elif args.model == 'clip512':
                    # Reshape x to [batch_size, channels, height, width]
                    encoder = model.encode_image
                    x = x.view(x.size(0), 3, 512, 512)
                    outputs = encoder(x)
                    logits = clf(outputs)
                elif args.model == 'clip':
                    # Reshape x to [batch_size, channels, height, width]
                    encoder = model.encode_image
                    x = x.view(x.size(0), 3, 224, 224)
                    outputs = encoder(x)
                    logits = clf(outputs)

                else:
                    encoder = torch.nn.Sequential(
                        *(list(model.children())[:-1] + [torch.nn.Flatten()]))
                    outputs = encoder(x)
                    logits = clf(outputs)


                if args.hessian_align:
                    # 'x' is the input the classifier, which is output of the encoder
                    loss_main, _, _, _ = loss_computer.exact_hessian_loss(logits, outputs, y, g, grad_alpha = args.grad_alpha, hess_beta = args.hess_beta)
                else:
                    loss_main = loss_computer.loss(logits, y, g, is_training)

                if is_training:
                    loss_main = loss_main / num_sub_batches
                    loss_main.backward()


            if is_training:
                if args.model == 'bert':
                    # loss_main.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                else:
                    # optimizer.zero_grad()
                    # loss.backward()
                    optimizer.step()


            if is_training and (batch_idx + 1) % log_every == 0:
                # breakpoint()
                csv_logger.log(epoch, batch_idx, loss_computer.get_stats(model, args))
                csv_logger.flush()
                loss_computer.log_stats(logger, is_training)
                loss_computer.reset_stats()

        if (not is_training) or loss_computer.batch_count > 0:
            # breakpoint()
            csv_logger.log(epoch, batch_idx, loss_computer.get_stats(model, args))
            csv_logger.flush()
            loss_computer.log_stats(logger, is_training)
            if is_training:
                loss_computer.reset_stats()

        del x, y, g, outputs, loss_main
        torch.cuda.empty_cache()


def train(model,clf, criterion, dataset,
          logger, train_csv_logger, val_csv_logger, test_csv_logger,
          args, epoch_offset, optimizer, scheduler):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model = model.cuda()
    # breakpoint()

    num_classes = 2
    if args.model == 'clip512':
        dummy_input = torch.randn(1, 3, 512, 512).cuda()
        encoder = model.encode_image
        with torch.no_grad():
            dummy_output = encoder(dummy_input)
    elif args.model == 'vits':
        dummy_input = torch.randn(1, 3, 224, 224).cuda()
        with torch.no_grad():
            dummy_output = model(dummy_input)
    elif args.model == 'clip':
        dummy_input = torch.randn(1, 3, 224, 224).cuda()
        encoder = model.encode_image
        with torch.no_grad():
            dummy_output = encoder(dummy_input)
    elif args.model == 'bert':
        dummy_input = torch.randint(0, 1000, (1, 128)).cuda()
        with torch.no_grad():
            dummy_output = model(dummy_input)
        num_classes = 3
    # else:
    #     # resnet50
    #     dummy_input = torch.randn(1, 3, 224, 224).cuda()
    #     encoder = torch.nn.Sequential(
    #                     *(list(model.children())[:-1] + [torch.nn.Flatten()]))
    #     with torch.no_grad():
    #         dummy_output = encoder(dummy_input)

    if clf is None:
        if 'resnet' in args.model:
            d = model.fc.in_features
            clf = LogisticRegression(d, num_classes).cuda()
        else:
            clf = LogisticRegression(dummy_output.size(-1), num_classes).cuda()


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
        # breakpoint()
        if optimizer is None:
            optimizer = torch.optim.AdamW(
                chain(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    filter(lambda p: p.requires_grad, clf.parameters())
                ),
                lr=args.lr,
                weight_decay=args.weight_decay
            )

        if args.scheduler:
            if scheduler is None:
                # breakpoint()
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
    # breakpoint()
    # for epoch in tqdm(range(epoch_offset, epoch_offset + args.n_epochs)):
    for epoch in tqdm(range(epoch_offset, args.n_epochs)):
        # with torch.autograd.profiler.profile(use_cuda=True) as prof:
        logger.write('\nEpoch [%d]:\n' % epoch)
        logger.write(f'Training:\n')

        run_epoch(
            epoch, model, clf, optimizer,
            dataset['train_loader'],
            train_loss_computer,
            logger, train_csv_logger, args,
            is_training=True,
            show_progress=args.show_progress,
            log_every=args.log_every,
            scheduler=scheduler)
        # breakpoint()
        logger.write(f'\nValidation:\n')
        val_loss_computer = LossComputer(
            criterion,
            is_robust=args.robust,
            dataset=dataset['val_data'],
            step_size=args.robust_step_size,
            alpha=args.alpha)
        run_epoch(
            epoch, model, clf, optimizer,
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
                epoch, model, clf, optimizer,
                dataset['test_loader'],
                test_loss_computer,
                None, test_csv_logger, args,
                is_training=False)

        # Inspect learning rates
        if (epoch + 1) % 1 == 0:
            for param_group in optimizer.param_groups:
                curr_lr = param_group['lr']
                # logger.write('Current lr: %f\n' % curr_lr)
                logger.write('Current lr: {}\n'.format(repr(curr_lr)))

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
            torch.save(clf, os.path.join(args.log_dir, 'last_clf.pth'))
            torch.save(optimizer, os.path.join(args.log_dir, 'last_optimizer.pth'))
            torch.save(scheduler, os.path.join(args.log_dir, 'last_scheduler.pth'))
        if args.save_best:
            curr_val_acc = min(val_loss_computer.avg_group_acc)
            curr_avg_val_acc = val_loss_computer.avg_acc

            logger.write(f'Current average validation accuracy: {curr_avg_val_acc}\n')
            logger.write(f'Current worst-group validation accuracy: {curr_val_acc}\n')
            if curr_val_acc > best_val_acc:
                best_val_acc = curr_val_acc
                torch.save(model, os.path.join(args.log_dir, 'best_model.pth'))
                torch.save(clf, os.path.join(args.log_dir, 'best_clf.pth'))
                torch.save(optimizer, os.path.join(args.log_dir, 'best_optimizer.pth'))
                torch.save(scheduler, os.path.join(args.log_dir, 'best_scheduler.pth'))
                logger.write(f'Best worst-group model saved at epoch {epoch}\n')
            if curr_avg_val_acc > best_avg_val_acc:
                best_avg_val_acc = curr_avg_val_acc
                torch.save(model, os.path.join(args.log_dir, 'best_avg_acc_model.pth'))
                torch.save(clf, os.path.join(args.log_dir, 'best_avg_acc_clf.pth'))
                torch.save(optimizer, os.path.join(args.log_dir, 'best_avg_acc_optimizer.pth'))
                torch.save(scheduler, os.path.join(args.log_dir, 'best_avg_acc_scheduler.pth'))
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
        # print(prof)
