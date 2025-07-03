import numpy as np

def corr(
    x: np.array, 
    y: np.array,
    rectified: bool = False
) -> float:
    """
    Calculate the correlation between two arrays.
    If `rectified` is True, it computes the rectified correlation,
    which only considers positive deviations from the mean.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x_std = np.std(x)
    y_std = np.std(y)
    if x_std == 0 or y_std == 0:
        return np.nan  # avoid divide-by-zero
    if rectified:
        return np.sum(np.maximum(0, x - x_mean) * np.maximum(0, y - y_mean)) / (len(x) * x_std * y_std)
    return np.sum((x - x_mean) * (y - y_mean)) / (len(x) * x_std * y_std)