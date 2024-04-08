import torch
import torch.nn.functional as F


def debug_hessian_differences(x, logits):
    # Define the original function for reference
    def hessian_original(x, logits):
        p = F.softmax(logits, dim=1).clone()[:, 1]  # probability for class 1
        scale_factor = p * (1 - p)  # Shape: [batch_size]
        scale_factor = scale_factor.view(-1, 1, 1)
        x_reshaped = x.unsqueeze(2)
        outer_product = torch.matmul(x_reshaped, x_reshaped.transpose(1, 2))
        hessian_w_class0 = torch.mean(scale_factor * outer_product, dim=0)
        hessian_w_class1 = -hessian_w_class0
        hessian_w2 = torch.stack([hessian_w_class0, hessian_w_class1])
        return hessian_w2

    # Define the modified function for reference
    def hessian(x, logits):
        softmax = torch.softmax(logits, dim=1)
        diag_terms = softmax * (1 - softmax)
        diag_terms = diag_terms.unsqueeze(-1)
        outer_product = torch.bmm(x.unsqueeze(2), x.unsqueeze(1))
        Hessian = torch.einsum('bni,bij->nij', diag_terms, outer_product)
        hessian_w = Hessian.mean(dim=0)
        return hessian_w

    # Compute Hessians using both functions
    hessian_original_result = hessian_original(x, logits)
    hessian_modified_result = hessian(x, logits)

    # Compare and print discrepancies
    print("Comparing final Hessian results:")
    print("Original Hessian shape:", hessian_original_result.shape)
    print("Modified Hessian shape:", hessian_modified_result.shape)

    if torch.allclose(hessian_original_result, hessian_modified_result):
        print("Final Hessian tensors are equivalent.")
    else:
        print("Final Hessian tensors differ.")

    # This is a simplified comparison focusing on final output.
    # For intermediate steps, you would insert similar comparison checks
    # right after each step within the original and modified functions
    # and print discrepancies. This might involve modifying the original
    # and modified functions to return intermediate results for comparison.


# Example usage:
batch_size, num_features, num_classes = 10, 4, 3
x = torch.randn(batch_size, num_features)
logits = torch.randn(batch_size, num_classes)

debug_hessian_differences(x, logits)
