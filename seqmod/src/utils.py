"""utils.py package that contains random functionality used throughout the sequence package."""

import random

import numpy as np

import matplotlib

import torch


def set_seed(seed: int = 42) -> bool:
    """
    Sets the random seed for all random number generators i.e. Python, NumPy, and PyTorch.
    Also takes out any randomness from cuda kernels.

    Parameters
    ----------
    seed : int, optional
        The seed value to set for all random number generators, by default 42.

    Returns
    -------
    bool
        Always returns True, indicating the random seeds and configurations were successfully set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False  # uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms.
    torch.backends.cudnn.enabled = False

    return True


def add_metric_plot(axes: matplotlib.axes._axes.Axes, X, y, legend: str, colour: str):
    """
    Plots a given metric on provided axes with specified legend and color.

    This function is intended for internal use to plot various metrics
    during model training or evaluation on a given set of axes from a matplotlib figure.

    Parameters
    ----------
    axes : matplotlib.axes._axes.Axes
        The axes on which to plot the metric.
    X : array-like
        The x-values of the points to plot.
    y : array-like
        The y-values of the points to plot.
    legend : str
        The legend label for the plotted line.
    colour : str
        The color of the plotted line.

    Returns
    -------
    None
    """
    axes.plot(X, y, label=legend, color=colour)
