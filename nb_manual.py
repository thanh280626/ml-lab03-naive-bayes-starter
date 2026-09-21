import math
import numpy as np


def compute_class_priors(y_train):
    """
    Mission 2: Tính xác suất tiên lượng Prior P(Y) cho từng lớp.
    Input: y_train (numpy array)
    Output: dict {class_label: prior_probability}
    """
    classes, counts = np.unique(y_train, return_counts=True)
    total_samples = len(y_train)
    
    priors = {}
    for cls, count in zip(classes, counts):
        priors[int(cls)] = float(count / total_samples)
        
    return priors


def gaussian_log_likelihood(x, mean, var, eps=1e-9):
    """
    Mission 3 & 5: Tính Gaussian Log-Likelihood log P(x_i | y) cho một đặc trưng.
    Input:
        x: giá trị đặc trưng
        mean: giá trị trung bình (theta)
        var: phương sai (sigma^2)
        eps: hằng số nhỏ tránh chia cho 0
    Output: giá trị log likelihood (float)
    """
    var = var + eps
    log_likelihood = -0.5 * math.log(2 * math.pi * var) - ((x - mean) ** 2) / (2 * var)
    return log_likelihood
