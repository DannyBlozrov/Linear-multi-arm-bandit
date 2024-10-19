import numpy as np
import scipy
import functions


import numpy as np
import scipy

def FW_optimal(arms: np.ndarray, indexes: np.ndarray, threshold,max_support=None):
    """
    :param arms: a matrix of current arm vectors
    :param indexes: indices of arms
    :param max_support: maximum number of non-zero elements allowed in pi
    :param tol: tolerance for small values to be considered as zero
    :return: a sparse vector of probabilities (0,1) for each arm, using optimization
    """
    d, k = np.shape(arms)
    max_support = max_support or (d * (d + 1)) // 2

    # Initial guess for pi
    pi_0 = np.zeros(k)
    for i in indexes:
        pi_0[i] = 1 / len(indexes)
    return FW(arms, pi_0, indexes, max_support)


def FW(arms, pi_0, indexes, max_support):
    """
    :param arms: matrix of size (d,k)
    :param pi_0: a vector of size (k) that is the distribution of each, an initial guess is just 1/len(indexes) at indexed slots
    :param indexes: a vector of size (l <= k) with the indexes of the arms we want
    :param max_support: maximum number of non-zero elements allowed in pi
    :return: a probability distribution pi that minimizes g(pi, arms, indexes)
    """
    (d, k) = arms.shape
    pi = pi_0

    while True:
        A_inverse = A_inv(pi, arms, indexes)
        all_norms = np.asarray([mah_norm(arms[:, i], A_inverse) for i in indexes])
        i_l = np.argmax(all_norms)  # returns an integer between 0 to k-1
        e_i_l = np.zeros(k)
        e_i_l[i_l] = 1

        # Define the objective function to minimize
        def f(gamma):
            input_vec = ((1 - gamma) * pi) + (gamma * e_i_l)
            return g(input_vec, arms, indexes)[1] * -1

        # Solve for optimal step size gamma_l
        gamma_l = scipy.optimize.minimize_scalar(f, bounds=(0, 1))
        solver = gamma_l['message']
        gamma_l = gamma_l['x']

        # Update the distribution pi
        pi = (1 - gamma_l) * pi + gamma_l * e_i_l

        # Enforce the sparsity constraint (limit non-zero entries to max_support)
        if np.count_nonzero(pi) > max_support:
            pi = sparsify(pi, max_support)

        A_inverse = A_inv(pi, arms, indexes)
        all_norms = np.asarray([mah_norm(arms[:, i], A_inverse) for i in indexes])

        # Check termination condition
        if np.max(all_norms) < 2 * d:
            pi = pi / np.sum(pi)  # Normalize pi to ensure it's a valid probability distribution
            return pi, solver


def sparsify(pi, max_support):
    """
    Sparsify the probability vector pi by keeping only the largest max_support entries
    :param pi: the probability distribution vector
    :param max_support: the maximum number of non-zero entries allowed in pi
    :return: a sparse version of pi with at most max_support non-zero entries
    """
    # Get indices of the largest max_support elements
    sorted_indices = np.argsort(-pi)  # Sort in descending order
    top_indices = sorted_indices[:max_support]

    # Create a new sparse pi with zeros everywhere except the top indices
    pi_sparse = np.zeros_like(pi)
    pi_sparse[top_indices] = pi[top_indices]

    # Normalize the sparse pi to ensure it's still a valid probability distribution
    pi_sparse /= np.sum(pi_sparse)

    return pi_sparse


def g(pi, arms, indexes):
    """
    :param pi: a (d) size vector of probabilities, a distribution
    :param arms: a (d,k) size matrix
    :return: the logdet of (sum_i pi_i x_i x_i ^T where x_i are the columns of the arms matrix
    """
    (d, k) = arms.shape
    A = np.zeros((d, d))
    for idx in indexes:
        A += (pi[idx] * np.outer(arms[:, idx], arms[:, idx]))
    sign, logabsdet = np.linalg.slogdet(A)
    return sign, logabsdet


def A_inv(pi: np.ndarray, arms: np.ndarray, indexes: np.ndarray):
    (d, k) = arms.shape
    A = np.zeros((d, d))
    for idx in indexes:
        A += pi[idx] * np.outer(arms[:, idx], arms[:, idx])
    A_inv = functions.invert_matrix(A)
    return A_inv


def mah_norm(x: np.ndarray, A: np.ndarray):
    """
    :param x: a (d) dimension vector
    :param A: a (d,d) PSD matrix
    :return: x^T A x
    """
    return x.T @ A @ x
