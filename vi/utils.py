import torch


def get_context_points(dim, n_context_points, context_points_bound):
    context_points_shape = (n_context_points, dim)
    context_points = (
        torch.rand(context_points_shape, dtype=torch.float32)
        * (context_points_bound[1] - context_points_bound[0])
        + context_points_bound[0]
    )
    return context_points


class StableSqrt(torch.autograd.Function):
    """
    Workaround to avoid the derivative of sqrt(0)
    This method returns sqrt(x) in its forward pass and in the backward pass
    it returns the gradient of sqrt(x) for all cases except for sqrt(0) where
    it returns the gradient 0
    """

    @staticmethod
    def forward(ctx, input):
        result = input.sqrt()
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        result = ctx.saved_tensors[0]
        grad = grad_output / (2.0 * result)
        grad[result == 0] = 0

        return grad
