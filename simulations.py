import math

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

def generate_linear_bandit_instance(k, d):
    """

    :param K: number of samples of a_i..a_k
    :param d: the dimension of each sample
    :return:a matrix of arm vectors of size dxk, theta star of size dx1
    """
    arm_vectors = np.random.rand(k, d)
    theta = np.random.rand(d).reshape(d,1)
    return arm_vectors.T, theta



# Function to perform dimensionality reduction
def projection_matrix(arm_vectors):
    """

    :param arm_vectors: a matrix of K vectors in dimension d'
    :return: an orthogonal spanning base matrix B with d rows and d' columns such that A'=B^T @ A
    """
    k,d = arm_vectors.shape
    Q, R = np.linalg.qr(arm_vectors.T)
    #Q is an orthogonal matrix
    #R is an upper triangular matrix
    nonzero_indices = np.where(~np.all(R == 0, axis=1))[0]
    projection_matrix = arm_vectors[nonzero_indices] #this is B^T before normalization
    return projection_matrix


def sequantial_choose(t,A):
    """

    :param t:  time slot to choose arm at
    :param A: the set of arms to choose from
    :return:
    """
    d,k = A.shape
    return A[t%k]

def get_reward(theta,arm):
    #loc is the mean, scale is the variance
    awgn = np.random.normal(loc = 0, scale = 1, size = 1)
    return theta@arm + awgn

def g_optimal(arm_matrix:np.ndarray,indexes:list):
    """

    :param arm_matrix: a matrix of current arm vectors
    :param indexes: indices of
    :return: a vector of probabilities (0,1) for each arm , using some optimization program
    """
    d,k = np.shape(arm_matrix)
    print(f"k={k},d={d}")
    num_vecs = len(indexes)
    print(f"arm matrix dims = {arm_matrix.shape}")
    print(f"indexes = {indexes}")
    #print(f"arm matrix = {arm_matrix}")
    V = lambda pi : sum(np.asarray([pi[i]*(arm_matrix[:,i].reshape((d,1))@(arm_matrix[:,i].reshape((d,1)).T))]) for i in indexes) #finding the matrix V for a distribution
    pi_0 = (1 / num_vecs) * np.ones((num_vecs)) #initial guess to find pi
    def g(pi):
        """
        :param pi: a probability distribution
        :return: the best arm in given distribution using a_i^T V(pi)^-1 a as the mahalobanius norm
        """
        V_pi:np.ndarray = V(pi)
        # print(f"Det = {np.linalg.det(V_pi)}")
        # print(f"V_pi = \n {V_pi}")
        return np.max([(arm_matrix[:,i].reshape((d,1))).T@np.linalg.inv(V_pi).reshape((d,d))@(arm_matrix[:,i].reshape((d,1))) for i in indexes])

    constraints = lambda v : np.sum(v) - 1 #sum of probabilities must equal 1
    constraints_dict = {'type':'eq','fun':constraints}
    pi_0 = (1/num_vecs) * np.ones((num_vecs))
    res = optimize.minimize(fun=g,bounds=[(0,1) for i in indexes],constraints = constraints_dict,x0 = pi_0)
    val = res['fun']
    pi = res['x']
    solver = res['message']
    return pi,solver

def simulate_fixed_budget(k=50, d=20, T=400,mean=0,sigma = 1, num_trials = 1):
    """

    :param k:  number of samples(arms)
    :param d: the dimension of each arm
    :param T: the budget for the simulation
    :param mean: the mean of the AWGN
    :param sigma: the standard deviation of the AWGN
    :param num_trials : number of trials to average the error
    :return:the optimal arm from the list of arms generated
    """
    original_arm_vectors, original_theta_star = generate_linear_bandit_instance(k,d)
    print(f"Arm Vectors =\n {original_arm_vectors}")
    curr_arms = original_arm_vectors
    curr_indexes = list(range(k))
    logd = math.ceil(math.log2(d))
    print(f"logd={logd}")
    curr_dim = d
    new_dim = d
    m = (T - np.min([k , (d*(d+1)/2)]) -sum([d/(2**r) for r in range(1,logd)]))/logd

    for r in range(1,logd+1):
        print(curr_arms.shape)
        new_dim = np.linalg.matrix_rank(curr_arms)
        print(f"new dim ={new_dim}")
        if (new_dim != curr_dim):
            (B,R) = np.linalg.qr(curr_arms,mode='reduced')
            print("SDSDSDSDS")
            print(f"k={k},d={d},B shape = {B.shape}")
            print(np.linalg.matrix_rank(B))
        if r == 1:
            pi,solver = g_optimal(curr_arms,curr_indexes)
            print(f"PI = {pi}r={r}") # vector of probabilities
        elif r != 1:
            pi,solver = g_optimal(curr_arms,curr_indexes)
            print(f"PI = {pi},r={r}")

        T_r_array = np.zeros((k))
        for i in curr_indexes:
            T_r_array[i] = np.ceil(pi[i]*m)
        Tr = np.sum(T_r_array)
        print(f"T_rs = {T_r_array}")
        print(f"Tr = {Tr}")
        V_r = sum(T_r_array[i]*(curr_arms[i]@curr_arms[i].T) for i in curr_indexes)
        print(f"V_r ={V_r}")
        newsum = np.zeros((curr_dim,1)) #from 19 in alg
        for idx in curr_indexes:
            for j in range(T_r_array[idx]):
                X_t = get_reward(original_theta_star,original_arm_vectors[idx])
                newsum += curr_arms[idx]*X_t
        Theta_r = np.linalg.inv(V_r)@newsum #this is Theta_r as in phase 19
        expected_rewards = np.zeros(k)
        for idx in curr_indexes:
            expected_rewards[idx] = np.dot(Theta_r,curr_arms[idx])
















if __name__ == "__main__":
    simulate_fixed_budget()



