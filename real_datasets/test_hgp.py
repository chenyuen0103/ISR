import torch
import pytest
from hisr import HISRClassifier
import copy
from torch.autograd import gradcheck
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


class HISRTester(HISRClassifier):
    def hgp_loss(self, model, x_batch, y_batch, envs_indices_batch, alpha=10e-5, beta=10e-5):
        torch.autograd.set_detect_anomaly(True)
        total_loss = torch.tensor(0.0, requires_grad=True)

        env_gradients = []
        env_hgp = []

        for env_idx in envs_indices_batch.unique():
            model.zero_grad()
            idx = (envs_indices_batch == env_idx).nonzero().squeeze()
            loss = self.loss_fn(model(x_batch[idx]).squeeze(), y_batch[idx].long())
            loss.backward(retain_graph=True)

            grads = [param.grad.clone().detach().requires_grad_(True) for param in model.parameters()]
            env_gradients.append(grads)

            grad_norm = torch.sqrt(sum(grad.norm(2) ** 2 for grad in grads))
            grad_norm.backward(retain_graph=True)

            grads_of_grad_norm = [param.grad.clone().detach() for param in model.parameters()]
            hessian_gradient_product = [grad_norm.clone().detach() * param for param in grads_of_grad_norm]
            env_hgp.append(hessian_gradient_product)

        avg_gradient = [torch.mean(torch.stack([grads[i] for grads in env_gradients]), dim=0) for i in
                        range(len(env_gradients[0]))]
        avg_hgp = [torch.mean(torch.stack([hess[i] for hess in env_hgp]), dim=0) for i in
                   range(len(env_hgp[0]))]

        for env_idx, (grads, hessian_gradient_product) in enumerate(zip(env_gradients, env_hgp)):
            idx = (envs_indices_batch == env_idx).nonzero().squeeze()
            loss = self.loss_fn(model(x_batch[idx]).squeeze(), y_batch[idx].long())

            grad_reg = sum((grad - avg_grad).norm(2) ** 2 for grad, avg_grad in zip(grads, avg_gradient))
            hgp_reg = sum((hgp - avg_hgp).norm(2) ** 2 for hgp, avg_hgp in zip(hessian_gradient_product, avg_hgp))

            total_loss = total_loss + (loss + alpha * hgp_reg + beta * grad_reg)

        n_unique_envs = len(envs_indices_batch.unique())
        total_loss = total_loss / n_unique_envs

        return total_loss






def test_loss_backward():
    # 1. Select a simple input
    x = torch.randn(16, 5, requires_grad=True)
    y = torch.randint(0, 2, (16,)).float()
    envs_indices = torch.randint(0, 4, (16,))
    num_classes = torch.unique(y).numel()
    # 2. Define the model and initialize an instance of the HISRTester class
    model = HISRClassifier.LogisticRegression(5, num_classes)
    original_model = copy.deepcopy(model)
    hisr = HISRTester(clf_type='LogisticRegression')

    # 3. Check the forward gradients using gradcheck
    x = x.double()
    model = model.double()
    # Flatten and combine all the model parameters into a single tensor
    model_parameters = [param for param in model.parameters()]

    def hgp_loss_wrapper(model, x_batch, y_batch, envs_indices_batch, alpha=10e-5, beta=10e-5):
        return hisr.hgp_loss(model, x_batch, y_batch, envs_indices_batch, alpha=alpha, beta=beta)


    # Flatten and concatenate model parameters and the actual inputs

    # Check the gradients for hgp_loss
    is_correct_hgp = gradcheck(hgp_loss_wrapper, (model, x, y.long(), envs_indices.long()), atol=1e-3, rtol=1e-3)
    assert is_correct_hgp, "Gradients of hgp_loss do not match!"


    # Select a specific environment index
    env_idx = envs_indices[0]
    idx = (envs_indices == env_idx).nonzero().squeeze()

    # Compute the hgp_loss for the specific environment
    target = y[idx].long()
    # loss = hisr.loss_fn(model(x[idx]).squeeze(), target, envs_indices)
    loss = hisr.hgp_loss(model, x, y, envs_indices)
    reference_state = copy.deepcopy(model.state_dict())
    # 2. Compute the analytical gradient using PyTorch's autograd
    loss.backward()
    analytical_gradients = [param.grad.clone() for param in model.parameters()]
    post_analytical_state = model.state_dict()
    assert all(torch.equal(reference_state[key], post_analytical_state[key]) for key in reference_state), \
        "Model state changed after computing analytical gradient"

    # Reset model to the original state
    model.load_state_dict(reference_state)
    # 3. Compute numerical gradients using the finite difference method
    epsilon = 10e-6
    numerical_gradients = []
    # model = copy.deepcopy(original_model)
    for param in model.parameters():
        param_grads = torch.zeros_like(param)
        for idx in range(param.nelement()):
            original_val = param.data.flatten()[idx].item()
            print(f"Orgininal parameter value: {original_val}", flush=True)
            param.data.flatten()[idx] = original_val + epsilon
            print(f"Perturbed parameter value (+): {param.data.flatten()[idx]}", flush=True)
            print(f"Model prediction for param+eps: {model(x[idx])}", flush=True)
            loss_plus_epsilon = hisr.hgp_loss(model, x, y, envs_indices)
            print(f"Loss of param + eps = {loss_plus_epsilon}", flush=True)
            param.data.flatten()[idx] = original_val - epsilon
            loss_minus_epsilon = hisr.hgp_loss(model, x, y, envs_indices)
            print(f"Perturbed parameter value -: {param.data.flatten()[idx]}", flush=True)
            print(f"Model prediction for param-eps: {model(x[idx])}", flush=True)
            print(f"Loss of param - eps = {loss_plus_epsilon}", flush=True)
            gradient = (loss_plus_epsilon - loss_minus_epsilon) / (2 * epsilon)
            param_grads.flatten()[idx] = gradient
            param.data.flatten()[idx] = original_val
            print(f"Analytical gradient: {param.grad.flatten()[idx]}", flush=True)
            print(f"Numerical gradient: {gradient}", flush=True)

        numerical_gradients.append(param_grads)
    post_numerical_state = model.state_dict()
    assert all(torch.equal(reference_state[key], post_numerical_state[key]) for key in reference_state), \
        "Model state changed after computing numerical gradient"

    # 4. Compare the analytical and numerical gradients
    threshold = 1e-4
    # for ag, ng in zip(analytical_gradients, numerical_gradients):
    #     assert torch.allclose(ag, ng, atol=threshold), f"Analytical and numerical gradients differ: {ag} vs {ng}"


@pytest.mark.parametrize("test_func", [test_loss_backward])
def run_test(test_func):
    test_func()


run_test(test_loss_backward)







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
