import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import gc
# from memory_profiler import profile
import cProfile




from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
import matplotlib.pyplot as plt

def to_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        return tensor
    else:
        raise ValueError(f"Unknown type: {type(tensor)}")


def feature_transform(Z: np.ndarray, u: np.ndarray, d_spu: int = 1, scale: float = 0) -> np.ndarray:
    scales = np.ones(Z.shape[1])
    scales[:d_spu] = scale
    # print(Z.shape, u.shape, scales.shape)
    Z = Z @ u @ np.diag(scales)
    return Z


def check_labels(labels) -> int:
    classes = np.unique(labels)
    n_classes = len(classes)
    assert np.all(classes == np.arange(n_classes)), f"Labels must be 0, 1, 2, ..., {n_classes - 1}"
    return n_classes


def estimate_means(zs, ys, gs, n_envs, n_classes) -> dict:
    Zs, Ys = {}, {}
    for e in range(n_envs):
        Zs[e] = zs[gs == e]
        Ys[e] = ys[gs == e]
    Mus = {}
    for label in range(n_classes):
        means = {}
        for e in range(n_envs):
            means[e] = np.mean(Zs[e][Ys[e] == label], axis=0)
        Mus[label] = np.vstack(list(means.values()))
    return Mus


def estimate_covs(zs, ys, gs, n_envs, n_classes) -> dict:
    Zs, Ys = {}, {}
    for e in range(n_envs):
        Zs[e] = zs[gs == e]
        Ys[e] = ys[gs == e]
    Covs = {label: {} for label in range(n_classes)}
    for label in range(n_classes):
        for e in range(n_envs):
            Covs[label][e] = np.cov(Zs[e][Ys[e] == label].T)
    return Covs


def check_clf(clf, n_classes):
    if isinstance(clf, LogisticRegression) or isinstance(clf, RidgeClassifier) or isinstance(clf, SGDClassifier):
        if n_classes == 2:
            assert 1 <= clf.coef_.shape[0] <= 2, f"The output dim of a binary classifier must be 1 or 2"
        else:
            assert clf.coef_.shape[0] == n_classes, f"The output dimension of the classifier must be {n_classes}."
        return clf
    elif isinstance(clf, torch.nn.Linear):
        weight = clf.weight.detach().data.cpu().numpy()
        bias = clf.bias.detach().data.cpu().numpy()
    elif isinstance(clf, dict):
        weight, bias = to_numpy(clf['weight']), to_numpy(clf['bias'])
    else:
        raise ValueError(f"Unknown classifier type: {type(clf)}")

    assert weight.shape[0] == len(
        bias), f"The output dimension of weight should match bias: {weight.shape[0]} vs {len(bias)}"
    sklearn_clf = LogisticRegression()
    sklearn_clf.n_classes = n_classes
    sklearn_clf.classes_ = np.arange(n_classes)
    sklearn_clf.coef_ = weight
    sklearn_clf.intercept_ = bias
    assert sklearn_clf.coef_.shape[0] == n_classes, f"The output dimension of the classifier must be {n_classes}."

    return sklearn_clf



class HISRClassifier:
    default_clf_kwargs = dict(C=1, max_iter=10000, random_state=0)

    def __init__(self, version: str = 'mean', hessian_approx_method = "exact", pca_dim: int = -1, d_spu: int = -1, spu_scale: float = 0,
                 chosen_class=None, clf_type: str = 'LogisticRegression', clf_kwargs: dict = None,
                 ):
        self.version = version
        self.hessian_approx_method = hessian_approx_method
        self.pca_dim = pca_dim
        self.d_spu = d_spu
        self.spu_scale = spu_scale

        self.clf_kwargs = HISRClassifier.default_clf_kwargs if clf_kwargs is None else clf_kwargs
        self.clf_type = clf_type
        self.chosen_class = chosen_class
        self.Us = {}  # stores computed projection matrices
        assert self.clf_type in ['LogisticRegression', 'RidgeClassifier', 'SGDClassifier'], \
            f"Unknown classifier type: {self.clf_type}"
        self.loss_fn = nn.CrossEntropyLoss() if self.clf_type == 'LogisticRegression' else nn.MSELoss()
        # self.loss_fn = nn.BCEWithLogitsLoss()
    def set_params(self, **params):
        for name, val in params.items():
            setattr(self, name, val)

    def fit(self, features, labels, envs, chosen_class: int = None, d_spu: int = None, given_clf=None,
            spu_scale: float = None):

        # estimate the stats (mean & cov) and fit a PCA if requested
        self.fit_data(features, labels, envs)

        if chosen_class is None:
            assert self.chosen_class is not None, "chosen_class must be specified if not given in the constructor"
            chosen_class = self.chosen_class

        if self.version == 'mean':
            self.fit_isr_mean(chosen_class=chosen_class, d_spu=d_spu)
        elif self.version == 'cov':
            self.fit_isr_cov(chosen_class=chosen_class, d_spu=d_spu)
        else:
            raise ValueError(f"Unknown ISR version: {self.version}")

        self.fit_clf(features, labels, given_clf=given_clf, spu_scale=spu_scale,alpha=10e-5, beta=10e-5)
        return self

    def fit_data(self, features, labels, envs, n_classes=None, n_envs=None):
        # estimate the mean and covariance of each class per environment
        self.n_classes = check_labels(labels)
        self.n_envs = check_labels(envs)
        if n_classes is not None: assert self.n_classes == n_classes
        if n_envs is not None: assert self.n_envs == n_envs

        # fit a PCA if requested
        if self.pca_dim > 0:
            self.pca = PCA(n_components=self.pca_dim).fit(features)
            features = self.pca.transform(features)
        else:
            self.pca = None
        self.means = estimate_means(features, labels, envs, self.n_envs, self.n_classes)
        self.covs = estimate_covs(features, labels, envs, self.n_envs, self.n_classes)
        return features

    def fit_isr_mean(self, chosen_class: int, d_spu: int = None):
        d_spu = self.d_spu if d_spu is None else d_spu
        assert d_spu < self.n_envs
        assert 0 <= chosen_class < self.n_classes
        # We project features into a subspace, and d_spu is the dimension of the subspace
        # Wew derive theoretically in the paper that the projection dimension of ISR-Mean
        # is at most n_envs-1
        if d_spu <= 0: self.d_spu = self.n_envs - 1

        key = ('mean', chosen_class, self.d_spu)
        if key in self.Us:
            return self.Us[key]

        # Estimate the empirical mean of each class

        # This PCA is just a helper function to obtain the projection matrix
        helper_pca = PCA(n_components=self.d_spu).fit(self.means[chosen_class])
        # The projection matrix has dimension (orig_dim, d_spu)
        # The SVD is just to pad the projection matrix with columns (the dimensions orthogonal
        # to the projection subspace) that makes the matrix a full-rank square matrix.
        U_proj = helper_pca.components_.T

        self.U = np.linalg.qr(U_proj, mode='complete')[0].real
        # The first d_spu dimensions of U correspond to spurious features, which we will
        # discard or reduce. The remaining dimensions are of the invariant feature subspace that
        # the algorithm identifies (not necessarily to be the real invariant features).

        # If we want to discard the spurious features, we can simply reduce the first d_spu
        # dimensions of U to zeros. However, this may hurt the performance of the algorithm sometimes,
        # so we can use the following strategy: rescale of the first d_spu dimensions with
        # factor between 0 and 1. This rescale factor is spu_scale that is chosen by the user.
        # print('\neig(U):', np.real(np.linalg.eigvals(self.U)))
        # print('singular vals:', s)
        self.Us[key] = self.U
        return self.U

    def fit_isr_cov(self, chosen_class: int, d_spu: int = None):
        self.d_spu = d_spu if d_spu is not None else self.d_spu
        assert self.d_spu > 0, "d_spu must be provided for ISR-Cov"
        # TODO: implement ISR-Cov for n_envs > 2
        assert self.n_envs == 2, "ISR-Cov is only implemented for binary env so far"

        key = ('cov', chosen_class, self.d_spu)
        if key in self.Us:
            return self.Us[key]

        env_pair = [0, 1]
        cov_0 = self.covs[chosen_class][env_pair[0]]
        cov_1 = self.covs[chosen_class][env_pair[1]]
        cov_diff = cov_1 - cov_0
        D = cov_diff.shape[0]

        # take square root of cov_diff such that the resulting matrix has non-negative eigenvalues
        # the largest d_spu eigenvalues correspond to the spurious feature subspace
        # we only need compute the eigenvectors of these d_spu dimensions (save computation cost)

        cov_sqr = cov_diff @ cov_diff
        w, U_proj = scipy.linalg.eigh(cov_sqr, subset_by_index=[D - self.d_spu, D - 1])
        assert w.min() >= 0
        order = np.flip(np.argsort(w).flatten())
        U_proj = U_proj[:, order]

        # trivially call SVD to fill the rest columns the (D-d_spu)-dim subspace orthogonal to the
        # spurious feature subspace

        self.U = np.linalg.svd(U_proj, full_matrices=True)[0]

        self.Us[key] = self.U

        return self.U



    def _flatten_grad(self, grads):
        flatten_grad = torch.cat([g.flatten() for g in grads])
        return flatten_grad

    def hessian(self,  x, logits):
        '''This function computes the hessian of the Cross Entropy with respect to the model parameters using the analytical form of hessian.'''
        # for params in model.parameters():
        #     params.requires_grad = True

        # logits = model(x)
        p = F.softmax(logits, dim=1).clone()[:, 1]  # probability for class 1

        # Compute the Hessian for each sample in the batch, then average
        batch_size = x.shape[0]
        hessian_list_class0 = [p[i] * (1 - p[i]) * torch.ger(x[i], x[i]) for i in range(batch_size)]

        hessian_w_class0 = sum(hessian_list_class0) / batch_size

        # Hessian for class 1 is just the negative of the Hessian for class 0
        hessian_w_class1 = -hessian_w_class0


        # Stacking the Hessians for both classes
        hessian_w = torch.stack([hessian_w_class0, hessian_w_class1])
        return hessian_w


    def gradient(self, x, logits, y):
        # for param in model.parameters():
        #     param.requires_grad = True

        # Compute logits and probabilities
        # logits = model(x)
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
        grad_w_class1 = torch.matmul((y_onehot[:, 1] - p[:, 1]).unsqueeze(0), x) / x.size(0)
        grad_w_class0 = torch.matmul((y_onehot[:, 0] - p[:, 0]).unsqueeze(0), x) / x.size(0)

        # Stack the gradients for both classes
        grad_w = torch.cat([grad_w_class1, grad_w_class0], dim=0)
        return grad_w

    def calc_hessian_diag(self, model, loss_grad, repeat=1000):
        diag = []
        gg = torch.cat([g.flatten() for g in loss_grad])
        assert gg.requires_grad, "Gradient tensor does not require gradient"
        for _ in range(repeat):
            z = 2*torch.randint_like(gg, high=2)-1
            loss = torch.dot(gg,z)
            assert loss.requires_grad, "Surrogate loss does not require gradient"
            Hz = torch.autograd.grad(loss, model.parameters(), retain_graph=True, create_graph=True)
            Hz = torch.cat([torch.flatten(g) for g in Hz])
            diag.append(z*Hz)
        return sum(diag)/len(diag)

    def hutchinson_loss(self, model, x_batch, y_batch, envs_indices_batch, alpha=10e-4, beta=10e-4):
        total_loss = torch.tensor(0.0, requires_grad=True)
        env_gradients = []
        env_hessian_diag = []
        combine_loss = self.loss_fn(model(x_batch).squeeze(), y_batch.long())
        average_gradient = torch.autograd.grad(combine_loss, model.parameters(), create_graph=True)
        average_Hg = self.calc_hessian_diag(model, average_gradient, repeat=150)
        for env_idx in envs_indices_batch.unique():
            model.zero_grad()

            idx = (envs_indices_batch == env_idx).nonzero().squeeze()
            loss = self.loss_fn(model(x_batch[idx]).squeeze(), y_batch[idx].long())
            assert loss.requires_grad, "Original loss does not require gradient"

            # get gradient of loss with respect to parameters
            loss.backward(retain_graph=True)

            # Gradient
            # grads = self.get_grads(model)
            # grads = [param.grad.clone().detach().requires_grad_(True) for param in model.parameters()]
            grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
            original_grads = [grad.clone().requires_grad_(True) for grad in grads]

            # flatten gradient
            flatten_grad = self._flatten_grad(grads)
            flatten_original_grad = self._flatten_grad(original_grads)
            env_gradients.append(flatten_grad)

            # ||Gradient||
            grad_norm = torch.sqrt(sum(grad.norm(2) ** 2 for grad in flatten_grad))
            grad_norm.backward(retain_graph=True)

            # Diag of Hessian, the arguments are: model, gradient of the current environment, number of mc steps to take
            Hg = self.calc_hessian_diag(model, flatten_original_grad, repeat = 150)
            env_hessian_diag.append(Hg)

            # average_Hg = torch.mean(torch.stack(env_hessian_diag), dim=0)
            # average_gradient = torch.mean(torch.stack(env_gradients), dim=0)


        for env_idx, (grad, hessian_diag) in enumerate(zip(env_gradients, env_hessian_diag)):
            idx = (envs_indices_batch == env_idx).nonzero().squeeze()
            loss = self.loss_fn(model(x_batch[idx]).squeeze(), y_batch[idx].long())

            grad_reg = sum((grad - avg_grad).norm(2) ** 2 for grad, avg_grad in zip(grads, average_gradient))
            hg_reg = sum((hg - avg_hg).norm(2) ** 2 for hg, avg_hg in zip(hessian_diag, average_Hg))

            total_loss = total_loss + (loss + alpha * grad_reg + beta * hg_reg)

        n_unique_envs = len(envs_indices_batch.unique())
        total_loss = total_loss / n_unique_envs

        return total_loss

    def hgp_loss(self, model, x_batch, y_batch, envs_indices_batch, alpha=10e-5, beta=10e-5):
        total_loss = torch.tensor(0.0, requires_grad=True)
        env_gradients = []
        env_hgp = []
        for env_idx in envs_indices_batch.unique():
            model.zero_grad()
            idx = (envs_indices_batch == env_idx).nonzero().squeeze()
            loss = self.loss_fn(model(x_batch[idx]).squeeze(), y_batch[idx].long())
            # get gradient of loss with respect to parameters
            loss.backward(retain_graph=True)

            # Gradient
            # grads = self.get_grads(model)
            grads = [param.grad.clone().detach().requires_grad_(True) for param in model.parameters()]
            env_gradients.append(grads)

            # ||Gradient||
            grad_norm = torch.sqrt(sum(grad.norm(2) ** 2 for grad in grads))
            grad_norm.backward(retain_graph=True)

            # gradient of ||Gradient||
            grads_of_grad_norm = [param.grad.clone().detach() for param in model.parameters()]

            # Approx. hessian-gradient-product
            hessian_gradient_product = [grad_norm.clone().detach() * param for param in grads_of_grad_norm]
            env_hgp.append(hessian_gradient_product)
        # Compute average gradient and hessian
        avg_gradient = [torch.mean(torch.stack([grads[i] for grads in env_gradients]), dim=0) for i in
                        range(len(env_gradients[0]))]
        avg_hgp = [torch.mean(torch.stack([hess[i] for hess in env_hgp]), dim=0) for i in
                   range(len(env_hgp[0]))]

        for env_idx, (grads, hessian_gradient_product) in enumerate(zip(env_gradients, env_hgp)):
            idx = (envs_indices_batch == env_idx).nonzero().squeeze()
            loss = self.loss_fn(model(x_batch[idx]).squeeze(), y_batch[idx].long())

            grad_reg = sum((grad - avg_grad).norm(2) ** 2 for grad, avg_grad in zip(grads, avg_gradient))
            hgp_reg = sum((hgp - avg_hgp).norm(2) ** 2 for hgp, avg_hgp in zip(hessian_gradient_product, avg_hgp))

            total_loss = total_loss + (loss + alpha * grad_reg + beta * hgp_reg)

        n_unique_envs = len(envs_indices_batch.unique())
        total_loss = total_loss / n_unique_envs

        return total_loss

    def compute_pytorch_hessian(self, model, x, y):
        batch_size = x.shape[0]
        for param in model.parameters():
            param.requires_grad = True

        logits = model(x).squeeze()

        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(logits, y.long())

        # First order gradients
        grads = torch.autograd.grad(loss, model.linear.weight, create_graph=True)[0]

        hessian = []
        for i in range(grads.size(1)):
            row = torch.autograd.grad(grads[0][i], model.linear.weight, create_graph=True, retain_graph=True)[0]
            hessian.append(row)

        return torch.stack(hessian).squeeze()



    def exact_hessian_loss(self, logits, x, y, envs_indices, alpha=10e-5, beta=10e-5):
        # for params in model.parameters():
        #     params.requires_grad = True
        total_loss = torch.tensor(0.0, requires_grad=True)
        env_gradients = []
        env_hessians = []
        # initial_state = model.state_dict()
        # logits = model(x)
        envs_indices_unique = envs_indices.unique()
        for env_idx in envs_indices_unique:
            # breakpoint()
            idx = (envs_indices == env_idx).nonzero().squeeze()
            if idx.numel() == 0:
                env_gradients.append(torch.zeros(1))
                env_hessians.append(torch.zeros(1))
                continue
            elif x[idx].dim() == 1:
                yhat_env = logits[idx].view(1, -1)
            else:
                yhat_env = logits[idx]
            # # Gradient and Hessian Computation assumes negative log loss
            # # grads = self.gradient(model, x[idx], y[idx])
            # get grads, hessian of loss with respect to parameters, and those to be backwarded later
            # loss.backward(retain_graph=True)
            # grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
            x_env = x[idx]
            y_env = y[idx]
            yhat_env = yhat_env[0] if isinstance(yhat_env, tuple) else yhat_env

            grads = self.gradient(x_env, yhat_env, y_env)
            # hessian = self.compute_pytorch_hessian(model, x[idx], y[idx])
            hessian = self.hessian(x_env, yhat_env)
            env_gradients.append(grads)
            env_hessians.append(hessian)


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
            if idx.numel() == 0:
                continue
            elif idx.dim() == 0:
                num_samples = 1
            else:
                num_samples = len(idx)
            y_env = y[idx]
            logits_env = logits[idx]
            env_fraction = len(idx) / len(envs_indices)
            loss = self.loss_fn(logits_env.squeeze(), y_env.long())
            # Compute the 2-norm of the difference between the gradient for this environment and the average gradient
            grad_diff_norm = torch.norm(grads[0] - avg_gradient, p=2)

            # Compute the Frobenius norm of the difference between the Hessian for this environment and the average Hessian
            hessian_diff = hessian - avg_hessian
            hessian_diff_norm = torch.norm(hessian_diff, p='fro')


            # grad_reg = sum((grad - avg_grad).norm(2) ** 2 for grad, avg_grad in zip(grads, avg_gradient))
            # hessian_reg = torch.trace((hessian - avg_hessian).t().matmul(hessian - avg_hessian))

            grad_reg = alpha * grad_diff_norm ** 2
            hessian_reg = beta * hessian_diff_norm ** 2

            total_loss = total_loss + (loss + hessian_reg + grad_reg) * env_fraction
            # total_loss = total_loss + loss
            erm_loss += loss * env_fraction
            grad_loss += alpha * grad_reg * env_fraction
            hess_loss += beta * hessian_reg * env_fraction

        # n_unique_envs = len(4)
        # print("Number of unique envs:", n_unique_envs)
        # total_loss = total_loss / n_unique_envs
        # erm_loss = erm_loss / n_unique_envs
        # hess_loss = hess_loss / n_unique_envs
        # grad_loss = grad_loss / n_unique_envs
        # print("Loss:", total_loss.item(), "; Hessian Reg:",  alpha * hessian_reg.item(), "; Gradient Reg:", beta * grad_reg.item())


        return total_loss, erm_loss, hess_loss, grad_loss
    # @profile
    def fit_hessian_clf(self, x, y, envs_indices, approx_type = "exact", alpha = 1e-4, beta = 1e-4):
        # Create the model based on the model type
        num_iterations = self.clf_kwargs['max_iter']
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # if approx_type == "exact":
        #     device = "cpu"
        num_classes = len(np.unique(y))
        if self.clf_type == 'LogisticRegression':
            model = self.LogisticRegression(x.shape[1], num_classes)
        elif self.clf_type == 'RidgeClassifier':
            model = self.RidgeRegression(x.shape[1])
        elif self.clf_type == 'SGDClassifier':
            model = self.SGDClassifier(x.shape[1], num_classes)
        else:
            raise ValueError(f"Unknown model type: {self.clf_type}")
        model = model.to(device)
        self.optimizer = optim.SGD(model.parameters(), lr=0.001)
        self.optimizer.zero_grad()

        # Transform the data to tensors
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x).float()
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y).float()
        if not isinstance(envs_indices, torch.Tensor):
            envs_indices = torch.tensor(envs_indices).int()

        dataset = TensorDataset(x, y, envs_indices)
        print("Starting training on", device)

        if approx_type in ['exact','control']:
            dataloader = DataLoader(dataset, batch_size=500, shuffle=True)
            pbar = tqdm(range(num_iterations), desc='Hessian iter', leave=False)
            for epoch in pbar:
                for x_batch, y_batch, envs_indices_batch in dataloader:
                    x_batch, y_batch, envs_indices_batch = x_batch.to(device), y_batch.to(device), envs_indices_batch.to(device)
                    if approx_type == "control":
                        total_loss = self.loss_fn(model(x_batch).squeeze(), y_batch.long())
                        erm_loss = total_loss.item
                        hess_penalty = 0
                        grad_penalty = 0
                    else:
                        logits = model(x_batch)
                        total_loss, erm_loss, hess_penalty, grad_penalty = self.exact_hessian_loss(logits, x_batch, y_batch, envs_indices_batch, alpha, beta)
                        erm_loss = erm_loss.item()
                        hess_penalty = hess_penalty.item()
                        grad_penalty = grad_penalty.item()


                    total_loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                if epoch % 500 == 0:
                    # info = {
                    #     "Loss": total_loss.item(),
                    #     "ERM Loss": erm_loss.item(),
                    #     "Hessian Reg": hess_penalty.item(),
                    #     "Gradient Reg": grad_penalty.item()
                    # }
                    print("Loss:", total_loss.item(), "; ERM Loss:", erm_loss, "; Hessian Reg:", hess_penalty, "; Gradient Reg:", grad_penalty)

            # self.clf = model.to('cpu')
            # self.clf = model
            # return self.clf
        else:
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
            for epoch in tqdm(range(num_iterations), desc = 'Hessian iter'):
                for x_batch, y_batch, envs_indices_batch in dataloader:
                    x_batch, y_batch, envs_indices_batch = x_batch.to(device), y_batch.to(device), envs_indices_batch.to(
                        device)
                    if approx_type == "HGP":
                        total_loss = self.hgp_loss(model, x_batch, y_batch, envs_indices_batch, alpha, beta)
                    elif approx_type == "HUT":
                        total_loss = self.hutchinson_loss(model, x_batch, y_batch, envs_indices_batch, alpha, beta)
                    else:
                        raise ValueError(f"Unknown hessian approximation type: {approx_type}")
                    total_loss.backward()

                    self.optimizer.step()
                    self.optimizer.zero_grad()  # Reset gradients to zero for the next iteration
                    torch.cuda.empty_cache()
        self.clf = model.to('cpu')
        # self.clf = model
        return self.clf





    def fit_clf(self, features=None, labels=None, envs = None, given_clf=None, sample_weight=None, hessian_approx = None, alpha = 10e-5, beta = 10e-5, spu_scale = None):
        if not hessian_approx:
            if given_clf is None:
                assert features is not None and labels is not None
                # remove 'gradient_hyperparam' and 'hessian_hyperparam' from clf_kwargs
                self.clf_kwargs.pop('gradient_hyperparam', None)
                self.clf_kwargs.pop('hessian_hyperparam', None)
                self.clf = getattr(linear_model, self.clf_type)(**self.clf_kwargs)
                features = self.transform(features, )
                self.clf.fit(features, labels, sample_weight=sample_weight)
            else:
                self.clf = check_clf(given_clf, n_classes=self.n_classes)
                self.clf.coef_ = self.clf.coef_ @ self.U
            return self.clf
        else:
            assert features is not None and labels is not None
            features = self.transform(features, )
            return self.fit_hessian_clf(features, labels, envs_indices=envs, approx_type=self.hessian_approx_method, alpha=alpha, beta=beta)

    def transform(self, features, ):
        if self.pca is not None:
            features = self.pca.transform(features)
        new_zs = feature_transform(features, u=self.U,
                                   d_spu=self.d_spu, scale=self.spu_scale)
        return new_zs

    def predict(self, features):
        zs = self.transform(features)

        return self.clf.predict(zs)

    def score(self, features, labels):
        zs = self.transform(features)

        return self.clf.score(zs, labels)

    def fit_transform(self, features, labels, envs, chosen_class, given_clf=None):
        self.fit(features, labels, envs, chosen_class, given_clf)
        return self.transform(features)

    # build custom module for logistic regression
    class LogisticRegression(torch.nn.Module):
        # build the constructor
        def __init__(self, n_inputs, n_outputs):
            super().__init__()
            self.linear = torch.nn.Linear(n_inputs, n_outputs)

        # make predictions
        def forward(self, x):
            # Just return the logits (raw scores). Softmax will be applied in the loss function.
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x).float()
            # print(x.shape)
            return self.linear(x)

        def predict(self, x):
            logits = self.forward(x)
            return torch.argmax(logits, dim=1)

        def score(self, X, y, sample_weight=None):
            return accuracy_score(y, self.predict(X), sample_weight=sample_weight)

    class RidgeRegression(torch.nn.Module):
        def __init__(self, n_inputs):
            super(RidgeClassifier, self).__init__()
            self.linear = torch.nn.Linear(n_inputs, 1)

        def forward(self, x):
            y_pred = self.linear(x)
            return y_pred

        def score(self, X, y, sample_weight=None):
            return accuracy_score(y, self.predict(X), sample_weight=sample_weight)

    class SGDClassifier(torch.nn.Module):
        def __init__(self, n_inputs):
            super(SGDClassifier, self).__init__()
            self.linear = torch.nn.Linear(n_inputs, 1)

        def forward(self, x):
            y_pred = torch.sigmoid(self.linear(x))
            return y_pred

        def score(self, X, y, sample_weight=None):
            return accuracy_score(y, self.predict(X), sample_weight=sample_weight)
