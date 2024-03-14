import numpy as np


def softmax(x, axis=1):
    """
    Implements a stabilized softmax along the correct index
    https://www.deeplearningbook.org/contents/numerical.html

    """
    x = np.atleast_2d(x)
    
    # Subtract the max for numerical stability
    shiftx = x - np.max(x, axis=axis, keepdims=True)
    
    # Calculate exp and then normalize for softmax
    exps = np.exp(shiftx)
    softmax = exps / np.sum(exps, axis=axis, keepdims=True)
    
    return softmax


def stable_log_sum(X):
    """
    Implement a stabilized log sum operation.

    Args:
        X: an array of shape (K, 2) for some K
    Returns:
        sum(log(sum(exp(X), axis=1))), avoiding underflow
    """
    # You can assume that this array is of shape (K, 2)
    assert X.shape[1] == 2 and len(X.shape) == 2

     # Use the max in each row to approximate log(exp(X[i, 0]) + exp(X[i, 1]))
    max_X = np.max(X, axis=1)

    # Sum over all rows to get the final result
    result = np.sum(max_X)

    return result
