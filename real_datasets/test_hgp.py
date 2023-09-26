import torch
import pytest
from hisr import HISRClassifier
import copy
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)



def test_hessian_gradient_product():
    # 1. Select a simple input
    x = torch.randn(32, 10)
    y = torch.randint(0, 2, (32,)).float()
    envs_indices = torch.randint(0, 4, (32,))

    # 2. Define the model
    model = HISRClassifier.LogisticRegression(10)
    original_model = copy.deepcopy(model)
    # 3. Initialize an instance of the HISR class
    hisr = HISRClassifier(clf_type='LogisticRegression')

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    # 4. Compute the hgp_loss
    loss = hisr.hgp_loss(model, x, y, envs_indices)

    # 5. Compute the analytical gradient using PyTorch's autograd
    loss.backward()
    analytical_gradients = [param.grad for param in model.parameters()]
    model = copy.deepcopy(original_model)
    # 6. Compute numerical gradients using the finite difference method
    epsilon = 1e-4
    numerical_gradients = []
    for param in model.parameters():
        param_grads = torch.zeros_like(param)
        for idx in range(param.nelement()):
            original_val = param.data.flatten()[idx].item()
            param.data.flatten()[idx] = original_val + epsilon
            loss_plus_epsilon = hisr.hgp_loss(model, x, y, envs_indices)
            param.data.flatten()[idx] = original_val - epsilon
            loss_minus_epsilon = hisr.hgp_loss(model, x, y, envs_indices)
            param.data.flatten()[idx] = original_val
            gradient = (loss_plus_epsilon - loss_minus_epsilon) / (2 * epsilon)
            param_grads.flatten()[idx] = gradient
        numerical_gradients.append(param_grads)

    # 7. Compare the analytical and numerical gradients
    threshold = 1e-4
    differences = [torch.abs(ag - ng) for ag, ng in zip(analytical_gradients, numerical_gradients)]
    are_close = [diff.lt(threshold).all() for diff in differences]

    # Assert that all differences are below the threshold
    for close in are_close:
        assert close.item(), "Difference between analytical and numerical gradient is above the threshold!"


# If this script is run directly, pytest will execute the test function
if __name__ == "__main__":
    pytest.main([__file__])
