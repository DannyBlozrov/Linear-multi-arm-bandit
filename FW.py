import numpy as np
import scipy

import functions


def FW_optimal(arms: np.ndarray, indexes: np.ndarray, max_support=None, tol=1e-7):
    """
    :param arm_matrix: a matrix of current arm vectors
    :param indexes: indices of arms
    :param max_support: maximum number of non-zero elements allowed in pi
    :param tol: tolerance for small values to be considered as zero
    :return: a sparse vector of probabilities (0,1) for each arm, using optimization
    """
    d, k = np.shape(arms)
    max_support = max_support or (d * (d + 1)) // 2

    # Constraints for the sum of probabilities to be 1
    constraints = lambda v: np.sum(v) - 1
    constraints_dict = {'type': 'eq', 'fun': constraints}

    # Initial guess for pi
    pi_0 = np.zeros(k)
    for i in indexes:
        pi_0[i] = 1 / len(indexes)
    return FW(arms,pi_0,indexes)
def FW(arms,pi_0,indexes):
    """

    :param arms: matrix of size (d,k)
    :param pi_0: a vector of size (k) that is the distribution of each, an initial guess is just 1/len(indexes) at indexed slots
    :param indexes a vector of size (l<= k ) with the indexes of the arms we want
    :return:a probability distribution pi that minimizes g(pi,arms,indexes)
    """
    (d,k) = arms.shape
    l = 1
    pi = pi_0
    while True:
        A_inverse = A_inv(pi,arms,indexes)
        all_norms = np.asarray([mah_norm(arms[:,i],A_inverse) for i in indexes])
        i_l = np.argmax(all_norms) #returns an integer between 0 to k-1
        e_i_l = np.zeros(k)
        e_i_l[i_l] = 1
        temp = lambda x : ((1-x)*pi) + (x*e_i_l)
        def f(gamma):
            # print(f"gamma = {gamma}")
            # print(f"pi = {[pi]}")
            # print(f"eil={e_i_l}")
            input_vec = ((1-gamma)*pi) + (gamma*e_i_l)
            return g(input_vec,arms,indexes)[1] * -1
        gamma_l= scipy.optimize.minimize_scalar(f,bounds=(0,1))
        solver = gamma_l['message']
        gamma_l = gamma_l['x']
        pi = (1-gamma_l)*pi + gamma_l * e_i_l

        #pi = pi / np.sum(pi)
        A_inverse = A_inv(pi, arms, indexes)
        all_norms = np.asarray([mah_norm(arms[:, i], A_inverse) for i in indexes])
        if np.max(all_norms) < 2 * d :
            print(f"pi = {pi}")
            pi = pi/np.sum(pi)
            return pi,solver





def g(pi , arms,indexes):
    """

    :param pi: a (d) size vector of probabilities, a distribution
    :param arms:a (d,k) size matrix
    :return: the logdet of (sum_i pi_i x_i x_i ^T where x_i are the columns of the arms matrix
    """
    (d,k) = arms.shape

    A = np.zeros((d,d))
    for idx in indexes:
        A += (pi[idx] * np.outer(arms[:,idx],arms[:,idx]))
    sign,logabsdet = np.linalg.slogdet(A)
    return sign,logabsdet

def A_inv(pi:np.ndarray,arms:np.ndarray,indexes:np.ndarray):
    (d, k) = arms.shape
    A = np.zeros((d, d))
    for idx in indexes:
        A += pi[idx] * np.outer(arms[:, idx], arms[:, idx])
    A_inv = functions.invert_matrix(A)
    return A_inv

def mah_norm(x:np.ndarray,A:np.ndarray):
    """
    :param x: a (d) dimension vector
    :param A: a (d,d) PSD matrix
    :return: x^T A x
    """
    print(f"A={A}")
    return x.T @ A @x



