import numpy as np
import torch
from skimage.metrics import structural_similarity


def accuracy_fn(y_true, y_pred):
    """Calculates accuracy between truth labels and predictions.

    Args:
        y_true (torch.Tensor): Truth labels for predictions.
        y_pred (torch.Tensor): Predictions to be compared to predictions.

    Returns:
        [torch.float]: Accuracy value between y_true and y_pred, e.g. 78.45
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc
def motion_blur_kernel(k):
    """
    Creates motion blur kernel with kernel size 'k'
    """
    kernel_motion_blur = np.zeros((k, k))

    for i in range(k):
        kernel_motion_blur[i, k - i - 1] = 1

        if i > 0:
            kernel_motion_blur[i, k - i] = 0.5
        if i < k - 1:
            kernel_motion_blur[i, k - i - 2] = 0.5
    kernel_motion_blur = kernel_motion_blur / np.sum(kernel_motion_blur)
    return kernel_motion_blur


def gaussian_noise(y, noise_level):
    r"""
    Samples and returns a realization of Gaussian noise with the same shape of y, with norm equal to noise_level * || y ||_2.
    """
    # Sample noise
    e = np.random.normal(loc=0, scale=1, size=y.shape)

    # Normalize
    e = e / np.linalg.norm(e.flatten())

    # Scale
    e = noise_level * np.linalg.norm(y.flatten()) * e
    return e


def ssim(x_pred, x_true):
    r"""
    Computes the SSIM between the computed/predicted reconstruction x_pred and the true solution x_true.
    """
    # Normalize in [0, 1] range
    x_pred = (x_pred - x_pred.min()) / (x_pred.max() - x_pred.min())
    x_true = (x_true - x_true.min()) / (x_true.max() - x_true.min())

    return structural_similarity(x_pred, x_true, data_range=1.0)


def psnr(x_pred, x_true):
    r"""
    Computes the PSNR between the computed/predicted reconstruction x_pred and the true solution x_true.
    """
    mse = np.mean(np.square(x_pred - x_true))
    if mse == 0:
        return 100
    return -10 * np.log10(mse)


def rel_err(x_pred, x_true):
    r"""
    Computes the relative error in 2-norm between the computed/predicted reconstruction x_pred and the true solution x_true.
    """
    return np.linalg.norm(x_pred.flatten() - x_true.flatten()) / np.linalg.norm(
        x_true.flatten()
    )