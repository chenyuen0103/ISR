import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import random
from torch.autograd import gradcheck
from real_datasets.hisr import HISRClassifier



def set_seed(seed_value=42):
    """Set seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

    # if you are using GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


set_seed(42)
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 2)  # Example linear layer

    def forward(self, x):
        return self.fc(x)


def gradient(model, x, y):
    for param in model.parameters():
        param.requires_grad = True

    # Compute logits and probabilities
    logits = model(x)
    p = F.softmax(logits, dim=1)

    # Reshape y to one-hot encoded form
    y_onehot = torch.zeros_like(p)
    y_onehot.scatter_(1, y.unsqueeze(1), 1)

    # Compute the gradient using the analytical form for each class
    grad_w_class1 = torch.matmul((y_onehot[:, 1] - p[:, 1]).unsqueeze(0), x) / x.size(0)
    grad_w_class0 = torch.matmul((y_onehot[:, 0] - p[:, 0]).unsqueeze(0), x) / x.size(0)

    # Stack the gradients for both classes
    grad_w = torch.cat([grad_w_class1, grad_w_class0], dim=0)

    return grad_w


def hessian(model, x):
    for params in model.parameters():
        params.requires_grad = True

    logits = model(x).squeeze()
    sigmoid_logits = torch.sigmoid(logits)

    # Compute the second derivative of the BCE loss with respect to logits
    entries = sigmoid_logits * (1 - sigmoid_logits)

    # Create a diagonal matrix
    D = torch.diag(entries.squeeze())

    # Compute the Hessian for a single-layer model
    hessian_w = torch.mm(x.t(), torch.mm(D, x)) / x.shape[0]

    return hessian_w

def compute_pytorch_hessian(model, x,y):
    for param in model.parameters():
        param.requires_grad = True

    # Forward pass
    # logits = model(x).squeeze()
    logits = model(x).squeeze()

    # criterion = torch.nn.BCEWithLogitsLoss()
    criterion = torch.nn.CrossEntropyLoss()


    # First order gradients
    grads = torch.autograd.grad(loss, model.linear.weight, create_graph=True)[0]
    sigmoid_logits = torch.sigmoid(logits)
    print("Sigmoid logits (PyTorch):", sigmoid_logits)

    hessian = []
    for i in range(grads.size(1)):
        row = torch.autograd.grad(grads[0][i], model.fc.weight, create_graph=True, retain_graph=True)[0]
        hessian.append(row)

    return torch.stack(hessian).squeeze()


def compute_numerical_gradient(model, criterion, x, y, epsilon=1e-6):
    # Store numerical gradients for each parameter
    numerical_gradients = []

    # Loop over each parameter in the model
    for param in model.parameters():
        # Create a tensor to store the numerical gradient for this parameter
        param_numerical_gradient = torch.zeros_like(param.data)

        # Loop over each element in the parameter tensor
        for idx in np.ndindex(param.data.size()):
            # Store original parameter value
            original_value = param.data[idx]

            # Perturb the parameter value by +epsilon
            param.data[idx] = original_value + epsilon
            loss_plus = criterion(model(x).squeeze(), y.float())

            # Perturb the parameter value by -epsilon
            param.data[idx] = original_value - epsilon
            loss_minus = criterion(model(x).squeeze(), y.float())

            # Compute the numerical gradient
            param_numerical_gradient[idx] = (loss_plus - loss_minus) / (2 * epsilon)

            # Reset the parameter to its original value
            param.data[idx] = original_value

        numerical_gradients.append(param_numerical_gradient)

    return numerical_gradients


# Your gradient and hessian functions here...

def test_gradient(epsilon=1e-6):
    hisr = HISRClassifier()
    # criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCEWithLogitsLoss()
    criterion = hisr.loss_fn
    num_samples = 5
    input_dim = 10
    output_dim = 2
    model = hisr.LogisticRegression(input_dim, output_dim)
    x = torch.randn(num_samples, input_dim)
    y = torch.randint(0, output_dim, (num_samples,))

    # Save initial model state
    initial_state = model.state_dict()

    # Analytical gradient
    analytical_gradient = hisr.gradient(model, x, y)
    # Reset model to its initial state
    model.load_state_dict(initial_state)
    model.zero_grad()
    loss = criterion(model(x).squeeze(), y.long())
    loss.backward()
    pytorch_grads = list(model.parameters())[0].grad


    print("analytical gradients", analytical_gradient)
    print("pytorch gradients", pytorch_grads)
    assert torch.allclose(pytorch_grads, analytical_gradient, atol=1e-4), "Pytorch vs Analytical gradient test failed!"
    print("Gradient test passed!")


def test_gradcheck():
    # Create a model instance
    model = SimpleModel()
    model = model.double()

    # Define the function to be checked
    # Define the function to be checked
    def loss_function(w, b, x, y):
        logits = torch.matmul(x, w.T) + b
        # criterion = torch.nn.BCEWithLogitsLoss()
        criterion = torch.nn.CrossEntropyLoss()
        return criterion(logits.squeeze(), y)

    # Create double precision tensors
    x = torch.randn(5, 10, dtype=torch.double)
    y = torch.randint(0, 2, (5,), dtype=torch.double)
    w = torch.randn(1, 10, dtype=torch.double, requires_grad=True)
    b = torch.randn(1, dtype=torch.double, requires_grad=True)

    # Check the gradients using gradcheck
    is_correct = gradcheck(loss_function, (w, b, x, y), eps=1e-6, atol=1e-4)

    # Assert the result
    assert is_correct, "Gradient check failed!"


def test_hessian():
    hisr = HISRClassifier()
    num_samples = 5
    input_dim = 10
    output_dim = 2
    model = hisr.LogisticRegression(input_dim, output_dim)
    x = torch.randn(num_samples, input_dim)
    y = torch.randint(0, output_dim, (num_samples,))

    # Save initial model state
    initial_state = model.state_dict()

    # Analytical hessian
    analytical_hessian = hisr.hessian(model, x)
    print("Analytical Hessian:\n", analytical_hessian)
    model.load_state_dict(initial_state)
    model.zero_grad()
    pytorch_hessian = hisr.compute_pytorch_hessian(model, x, y)
    print("PyTorch Hessian:\n", pytorch_hessian)
    pytorch_hessian = pytorch_hessian.permute(1, 0, 2)
    diff = torch.abs(analytical_hessian - pytorch_hessian)
    max_diff = torch.max(diff).item()
    print("Max difference:", max_diff)

    assert torch.allclose(pytorch_hessian, analytical_hessian, atol=1e-4), "Hessian test failed!"

    print("Hessian test passed!")



def test_loss():
    hisr = HISRClassifier()
    # criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCEWithLogitsLoss()
    criterion = hisr.loss_fn
    num_samples = 5
    input_dim = 10
    output_dim = 2
    model = hisr.LogisticRegression(input_dim, output_dim)
    x = torch.randn(num_samples, input_dim)
    y = torch.randint(0, output_dim, (num_samples,))

    # Save initial model state
    initial_state = model.state_dict()

    y = y.long()
    logits = model(x).squeeze().long()
    loss = criterion(logits, y)
    # loss = criterion(model(x).squeeze().long(), y)
    grad_w, grad_b = hisr.gradient(model, x, y)
    hessian_matrix = hisr.hessian(model, x)

    grad_params = torch.cat([grad_w.flatten()])
    hess_norm = torch.norm(hessian_matrix, p='fro')
    grad_norm = torch.norm(grad_params, p=2)

    # Total loss
    total_loss = loss + hess_norm + grad_norm
    print("Analytical total loss:", total_loss)
    for name, param in model.named_parameters():
        print(name, param.grad)
    # Zero out gradients and perform backward
    optimizer.zero_grad()
    total_loss.backward()
    analytical_grads = {}
    for name, param in model.named_parameters():
        analytical_grads[name] = param.grad.clone()




    model.load_state_dict(initial_state)
    model.zero_grad()
    loss = criterion(model(x).squeeze(), y.long())
    loss.backward(retain_graph=True)

    pytorch_grads = list(model.parameters())[0].grad
    model.load_state_dict(initial_state)
    model.zero_grad()
    pytorch_hessian = compute_pytorch_hessian(model, x, y)

    pytorch_hess_norm = torch.norm(pytorch_hessian, p='fro')
    pytorch_grad_norm = torch.norm(pytorch_grads, p=2)
    pytorch_total_loss = loss + pytorch_hess_norm + pytorch_grad_norm
    pytorch_total_loss.backward()

    print("PyTorch total loss:", pytorch_total_loss)
    pytorch_grads = {}
    for name, param in model.named_parameters():
        pytorch_grads[name] = param.grad.clone()

    # 3. Compare the Gradients
    for name, grad in analytical_grads.items():
        assert torch.allclose(grad, pytorch_grads[name], atol=1e-4), f"Gradient mismatch for {name}!"
        print(grad, pytorch_grads[name])