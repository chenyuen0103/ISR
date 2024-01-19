import torch
import torch.nn.functional as F

# Set the default CUDA device to GPU 2
# torch.cuda.set_device(3)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class LossComputer:
    def __init__(self, criterion, is_robust, dataset, alpha=None, gamma=0.1, adj=None, min_var_weight=0, step_size=0.01,
                 normalize_loss=False, btl=False):
        self.criterion = criterion
        self.is_robust = is_robust
        self.gamma = gamma
        self.alpha = alpha
        self.min_var_weight = min_var_weight
        self.step_size = step_size
        self.normalize_loss = normalize_loss
        self.btl = btl

        self.n_groups = dataset.n_groups
        self.group_counts = dataset.group_counts().cuda()
        # self.group_counts = dataset.group_counts().to(device)
        self.group_frac = self.group_counts / self.group_counts.sum()
        self.group_str = dataset.group_str

        if adj is not None:
            self.adj = torch.from_numpy(adj).float().cuda()
            # self.adj = torch.from_numpy(adj).float().to(device)
        else:
            self.adj = torch.zeros(self.n_groups).float().cuda()
            # self.adj = torch.zeros(self.n_groups).float().to(device)
        if is_robust:
            assert alpha, 'alpha must be specified'

        # quantities maintained throughout training
        self.adv_probs = torch.ones(self.n_groups).cuda() / self.n_groups
        self.exp_avg_loss = torch.zeros(self.n_groups).cuda()
        self.exp_avg_initialized = torch.zeros(self.n_groups).byte().cuda()

        # self.adv_probs = torch.ones(self.n_groups).to(device) / self.n_groups
        # self.exp_avg_loss = torch.zeros(self.n_groups).to(device)
        # self.exp_avg_initialized = torch.zeros(self.n_groups).byte().to(device)

        self.reset_stats()

    def loss(self, yhat, y, group_idx=None, is_training=False):
        # compute per-sample and per-group losses
        per_sample_losses = self.criterion(yhat, y)
        group_loss, group_count = self.compute_group_avg(per_sample_losses, group_idx)
        group_acc, group_count = self.compute_group_avg((torch.argmax(yhat, 1) == y).float(), group_idx)

        # update historical losses
        self.update_exp_avg_loss(group_loss, group_count)

        # compute overall loss
        if self.is_robust and not self.btl:
            actual_loss, weights = self.compute_robust_loss(group_loss, group_count)
        elif self.is_robust and self.btl:
            actual_loss, weights = self.compute_robust_loss_btl(group_loss, group_count)
        else:
            actual_loss = per_sample_losses.mean()
            weights = None

        # update stats
        self.update_stats(actual_loss, group_loss, group_acc, group_count, weights)

        return actual_loss

    def compute_robust_loss(self, group_loss, group_count):
        adjusted_loss = group_loss
        if torch.all(self.adj > 0):
            adjusted_loss += self.adj / torch.sqrt(self.group_counts)
        if self.normalize_loss:
            adjusted_loss = adjusted_loss / (adjusted_loss.sum())
        self.adv_probs = self.adv_probs * torch.exp(self.step_size * adjusted_loss.data)
        self.adv_probs = self.adv_probs / (self.adv_probs.sum())

        robust_loss = group_loss @ self.adv_probs
        return robust_loss, self.adv_probs

    def compute_robust_loss_btl(self, group_loss, group_count):
        adjusted_loss = self.exp_avg_loss + self.adj / torch.sqrt(self.group_counts)
        return self.compute_robust_loss_greedy(group_loss, adjusted_loss)

    def compute_robust_loss_greedy(self, group_loss, ref_loss):
        sorted_idx = ref_loss.sort(descending=True)[1]
        sorted_loss = group_loss[sorted_idx]
        sorted_frac = self.group_frac[sorted_idx]

        mask = torch.cumsum(sorted_frac, dim=0) <= self.alpha
        weights = mask.float() * sorted_frac / self.alpha
        last_idx = mask.sum()
        weights[last_idx] = 1 - weights.sum()
        weights = sorted_frac * self.min_var_weight + weights * (1 - self.min_var_weight)

        robust_loss = sorted_loss @ weights

        # sort the weights back
        _, unsort_idx = sorted_idx.sort()
        unsorted_weights = weights[unsort_idx]
        return robust_loss, unsorted_weights
    def compute_pytorch_hessian(self, model, x, y):
        batch_size = x.shape[0]
        for param in model.parameters():
            param.requires_grad = True

        logits = model(x)
        logits = logits[0] if isinstance(logits, tuple) else logits
        logits = logits.squeeze()
        loss = self.criterion2(logits, y.long())

        # First order gradients
        grads = torch.autograd.grad(loss, model.linear.weight, create_graph=True)[0]

        hessian = []
        for i in range(grads.size(1)):
            row = torch.autograd.grad(grads[0][i], model.linear.weight, create_graph=True, retain_graph=True)[0]
            hessian.append(row)

        return torch.stack(hessian).squeeze()

    def hessian(self, model, x):
        '''This function computes the hessian of the Cross Entropy with respect to the model parameters using the analytical form of hessian.'''
        for params in model.parameters():
            params.requires_grad = True

        logits = model(x)
        logits = logits[0] if isinstance(logits, tuple) else logits
        p = F.softmax(logits, dim=1)[:, 1]  # probability for class 1

        # Compute the Hessian for each sample in the batch, then average
        batch_size = x.shape[0]
        breakpoint()
        hessian_list_class0 = [p[i] * (1 - p[i]) * torch.ger(x[i].flatten(), x[i].flatten()) for i in range(batch_size)]

        hessian_w_class0 = sum(hessian_list_class0) / batch_size

        # Hessian for class 1 is just the negative of the Hessian for class 0
        hessian_w_class1 = -hessian_w_class0


        # Stacking the Hessians for both classes
        hessian_w = torch.stack([hessian_w_class0, hessian_w_class1])
        return hessian_w


    def gradient(self,model, x, y):
        for param in model.parameters():
            param.requires_grad = True

        # Compute logits and probabilities
        logits = model(x)
        logits = logits[0] if isinstance(logits, tuple) else logits
        if logits.dim() == 1:
            p = F.softmax(logits, dim=0)
        else:
            p = F.softmax(logits, dim=1)


        y_onehot = torch.zeros_like(p)
        if len(y.shape) == 0:
            y = y.unsqueeze(0)
        # Check if p is 1D and if so, reshape it to 2D
        if len(p.shape) == 1:
            p = p.unsqueeze(0)

        # Ensure y_onehot is 2D: (batch_size, num_classes)
        num_classes = 2  # or whatever the number of classes is in your problem
        if len(y_onehot.shape) == 1:
            y_onehot = y_onehot.unsqueeze(0)

        # Now, scatter should work without errors
        y_onehot = y_onehot.scatter(1, y.unsqueeze(1).long(), 1)
        try:
            y_onehot = y_onehot.scatter(1, y.unsqueeze(1).long(), 1)
        except:
            y_onehot = y_onehot.scatter(1, y.unsqueeze(1), 1)

        # print("y.shape:", y.shape)
        # print("y_onehot.shape:", y_onehot.shape)

        # Compute the gradient using the analytical form for each class
        x_flattened = x.view(x.size(0), -1)
        weights1 = (y_onehot[:, 1] - p[:, 1]).unsqueeze(1)
        weights0 = (y_onehot[:, 0] - p[:, 0]).unsqueeze(1)

        # Perform matrix multiplication
        # The result will have the shape [1, 3 * 224 * 224]
        grad_w_class1 = torch.matmul(weights1.T, x_flattened)
        # grad_w_class1 = torch.matmul((y_onehot[:, 1] - p[:, 1]).unsqueeze(1), x) / x.size(0)
        grad_w_class0 = torch.matmul(weights0.T, x_flattened) / x.size(0)

        # Stack the gradients for both classes
        grad_w = torch.cat([grad_w_class1, grad_w_class0], dim=0)
        return grad_w

    def exact_hessian_loss(self, model, x, y, envs_indices, alpha=10e-5, beta=10e-5):
        total_loss = torch.tensor(0.0, requires_grad=True)
        self.criterion2 = torch.nn.CrossEntropyLoss()
        env_gradients = []
        env_hessians = []
        initial_state = model.state_dict()
        for env_idx in envs_indices.unique():
            model.zero_grad()
            idx = (envs_indices == env_idx).nonzero().squeeze()
            # loss = self.criterion(model(x[idx]).squeeze(), y[idx].long())
            # breakpoint()
            yhat = model(x[idx])
            # Assuming the first element of the tuple is the output you need
            main_output = yhat[0] if isinstance(yhat, tuple) else yhat
            # per_sample_loss = self.criterion(main_output, y[idx].long())
            # loss = per_sample_loss.mean()
            loss = self.criterion2(main_output, y[idx].long())

            # # Gradient and Hessian Computation assumes negative log loss
            # # grads = self.gradient(model, x[idx], y[idx])
            # get grads, hessian of loss with respect to parameters, and those to be backwarded later
            # breakpoint()
            loss.backward(retain_graph=True)
            # grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
            grads = self.gradient(model, x[idx], y[idx])
            # hessian = self.compute_pytorch_hessian(model, x[idx], y[idx])
            hessian = self.compute_pytorch_hessian(model, x[idx], y[idx])
            env_gradients.append(grads)
            env_hessians.append(hessian)

            model.load_state_dict(initial_state)
            model.zero_grad()

        # Compute average gradient and hessian
        # avg_gradient = [torch.mean(torch.stack([grads[i] for grads in env_gradients]), dim=0) for i in
        #                 range(len(env_gradients[0]))]
        weight_gradients = [g[0] for g in env_gradients]
        avg_gradient = torch.mean(torch.stack(weight_gradients), dim=0)

        # avg_gradient = torch.mean(torch.stack(env_gradients), dim=0)
        avg_hessian = torch.mean(torch.stack(env_hessians), dim=0)

        erm_loss = 0
        hess_loss = 0
        grad_loss = 0
        for env_idx, (grads, hessian) in enumerate(zip(env_gradients, env_hessians)):
            idx = (envs_indices == env_idx).nonzero().squeeze()
            loss = self.criterion(model(x[idx]).squeeze(), y[idx].long())
            # Compute the 2-norm of the difference between the gradient for this environment and the average gradient
            grad_diff_norm = torch.norm(grads[0] - avg_gradient, p=2)

            # Compute the Frobenius norm of the difference between the Hessian for this environment and the average Hessian
            hessian_diff = hessian - avg_hessian
            hessian_diff_norm = torch.norm(hessian_diff, p='fro')


            # grad_reg = sum((grad - avg_grad).norm(2) ** 2 for grad, avg_grad in zip(grads, avg_gradient))
            # hessian_reg = torch.trace((hessian - avg_hessian).t().matmul(hessian - avg_hessian))

            grad_reg = alpha * grad_diff_norm ** 2
            hessian_reg = beta * hessian_diff_norm ** 2

            total_loss = total_loss + (loss + hessian_reg + grad_reg)
            # total_loss = total_loss + loss
            erm_loss += loss
            grad_loss += alpha * grad_reg
            hess_loss += beta * hessian_reg

        n_unique_envs = len(envs_indices.unique())
        # print("Number of unique envs:", n_unique_envs)
        total_loss = total_loss / n_unique_envs
        erm_loss = erm_loss / n_unique_envs
        hess_loss = hess_loss / n_unique_envs
        grad_loss = grad_loss / n_unique_envs
        # print("Loss:", total_loss.item(), "; Hessian Reg:",  alpha * hessian_reg.item(), "; Gradient Reg:", beta * grad_reg.item())
        del grads
        del hessian
        del env_gradients
        del env_hessians
        torch.cuda.empty_cache()

        return total_loss, erm_loss, hess_loss, grad_loss

    def compute_group_avg(self, losses, group_idx):
        # compute observed counts and mean loss for each group
        group_map = (group_idx == torch.arange(self.n_groups).unsqueeze(1).long().cuda()).float()
        # group_map = (group_idx == torch.arange(self.n_groups).unsqueeze(1).long().to(device)).float()
        group_count = group_map.sum(1)
        group_denom = group_count + (group_count == 0).float()  # avoid nans
        group_loss = (group_map @ losses.view(-1)) / group_denom
        return group_loss, group_count

    def update_exp_avg_loss(self, group_loss, group_count):
        prev_weights = (1 - self.gamma * (group_count > 0).float()) * (self.exp_avg_initialized > 0).float()
        curr_weights = 1 - prev_weights
        self.exp_avg_loss = self.exp_avg_loss * prev_weights + group_loss * curr_weights
        self.exp_avg_initialized = (self.exp_avg_initialized > 0) + (group_count > 0)

    def reset_stats(self):
        self.processed_data_counts = torch.zeros(self.n_groups).cuda()
        self.update_data_counts = torch.zeros(self.n_groups).cuda()
        self.update_batch_counts = torch.zeros(self.n_groups).cuda()
        self.avg_group_loss = torch.zeros(self.n_groups).cuda()
        self.avg_group_acc = torch.zeros(self.n_groups).cuda()

        # self.processed_data_counts = torch.zeros(self.n_groups).to(device)
        # self.update_data_counts = torch.zeros(self.n_groups).to(device)
        # self.update_batch_counts = torch.zeros(self.n_groups).to(device)
        # self.avg_group_loss = torch.zeros(self.n_groups).to(device)
        # self.avg_group_acc = torch.zeros(self.n_groups).to(device)

        self.avg_per_sample_loss = 0.
        self.avg_actual_loss = 0.
        self.avg_acc = 0.
        self.batch_count = 0.

    def update_stats(self, actual_loss, group_loss, group_acc, group_count, weights=None):
        # avg group loss
        denom = self.processed_data_counts + group_count
        denom += (denom == 0).float()
        prev_weight = self.processed_data_counts / denom
        curr_weight = group_count / denom
        self.avg_group_loss = prev_weight * self.avg_group_loss + curr_weight * group_loss

        # avg group acc
        self.avg_group_acc = prev_weight * self.avg_group_acc + curr_weight * group_acc

        # batch-wise average actual loss
        denom = self.batch_count + 1
        self.avg_actual_loss = (self.batch_count / denom) * self.avg_actual_loss + (1 / denom) * actual_loss

        # counts
        self.processed_data_counts += group_count
        if self.is_robust:
            self.update_data_counts += group_count * ((weights > 0).float())
            self.update_batch_counts += ((group_count * weights) > 0).float()
        else:
            self.update_data_counts += group_count
            self.update_batch_counts += (group_count > 0).float()
        self.batch_count += 1

        # avg per-sample quantities
        group_frac = self.processed_data_counts / (self.processed_data_counts.sum())
        self.avg_per_sample_loss = group_frac @ self.avg_group_loss
        self.avg_acc = group_frac @ self.avg_group_acc

    def get_model_stats(self, model, args, stats_dict):
        model_norm_sq = 0.
        for param in model.parameters():
            model_norm_sq += torch.norm(param) ** 2
        stats_dict['model_norm_sq'] = model_norm_sq.item()
        stats_dict['reg_loss'] = args.weight_decay / 2 * model_norm_sq.item()
        return stats_dict

    def get_stats(self, model=None, args=None):
        stats_dict = {}
        for idx in range(self.n_groups):
            stats_dict[f'avg_loss_group:{idx}'] = self.avg_group_loss[idx].item()
            stats_dict[f'exp_avg_loss_group:{idx}'] = self.exp_avg_loss[idx].item()
            stats_dict[f'avg_acc_group:{idx}'] = self.avg_group_acc[idx].item()
            stats_dict[f'processed_data_count_group:{idx}'] = self.processed_data_counts[idx].item()
            stats_dict[f'update_data_count_group:{idx}'] = self.update_data_counts[idx].item()
            stats_dict[f'update_batch_count_group:{idx}'] = self.update_batch_counts[idx].item()

        stats_dict['avg_actual_loss'] = self.avg_actual_loss.item()
        stats_dict['avg_per_sample_loss'] = self.avg_per_sample_loss.item()
        stats_dict['avg_acc'] = self.avg_acc.item()

        # Model stats
        if model is not None:
            assert args is not None
            stats_dict = self.get_model_stats(model, args, stats_dict)

        return stats_dict

    def log_stats(self, logger, is_training):
        if logger is None:
            return

        logger.write(f'Average incurred loss: {self.avg_per_sample_loss.item():.3f}  \n')
        logger.write(f'Average sample loss: {self.avg_actual_loss.item():.3f}  \n')
        logger.write(f'Average acc: {self.avg_acc.item():.3f}  \n')
        for group_idx in range(self.n_groups):
            logger.write(
                f'  {self.group_str(group_idx)}  '
                f'[n = {int(self.processed_data_counts[group_idx])}]:\t'
                f'loss = {self.avg_group_loss[group_idx]:.3f}  '
                f'exp loss = {self.exp_avg_loss[group_idx]:.3f}  '
                f'adjusted loss = {self.exp_avg_loss[group_idx] + self.adj[group_idx] / torch.sqrt(self.group_counts)[group_idx]:.3f}  '
                f'adv prob = {self.adv_probs[group_idx]:3f}   '
                f'acc = {self.avg_group_acc[group_idx]:.3f}\n')
        logger.flush()
