from skimage.metrics import structural_similarity as ssim_metric
import torch


def mse(x: torch.Tensor, y: torch.Tensor) -> float:
    """
    Computes the Mean Squared Error (MSE) between two tensors
    :param x: First tensor
    :param y: Second tensor
    :return: Mean Squared Error as a float
    """
    return ((x - y) ** 2).mean().item()

def ssim(x: torch.Tensor, y: torch.Tensor) -> float:
    """
    Computes the Structural Similarity Index (SSIM) between two tensors
    :param x: First tensor
    :param y: Second tensor
    :return: SSIM as a float
    """
    x_np = x.squeeze().cpu().numpy()
    y_np = y.squeeze().cpu().numpy()
    return ssim_metric(x_np, y_np, data_range=x_np.max() - x_np.min())
