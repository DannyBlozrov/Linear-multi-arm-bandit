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
    theta = np.random.rand(d).reshape(d, 1)
    return arm_vectors.T, theta


# Function to perform dimensionality reduction
def projection_matrix(arm_vectors):
    """

    :param arm_vectors: a matrix of K vectors in dimension d'
    :return: an orthogonal spanning base matrix B with d rows and d' columns such that A'=B^T @ A
    """
    k, d = arm_vectors.shape
    Q, R = np.linalg.qr(arm_vectors.T)
    # Q is an orthogonal matrix
    # R is an upper triangular matrix
    nonzero_indices = np.where(~np.all(R == 0, axis=1))[0]
    projection_matrix = arm_vectors[nonzero_indices]  # this is B^T before normalization
    return projection_matrix


def sequantial_choose(t, A):
    """

    :param t:  time slot to choose arm at
    :param A: the set of arms to choose from
    :return:
    """
    d, k = A.shape
    return A[t % k]


def get_reward(theta, arm):

    # loc is the mean, scale is the variance
    armdim = arm.shape[0]
    arm = np.reshape(arm,(armdim,1))
    awgn = np.random.normal(loc=0, scale=1, size=1)
    return theta.T @ arm + awgn


def get_real_reward(theta, arm):
    # Ensure arm is a column vector
    arm = arm.reshape(-1, 1)  # d x 1
    # Perform the matrix multiplication
    return (theta.T @ arm).item()  # Result is a scalar


def best_reward_vec(arms, theta):
    max_reward = -np.inf
    best_arm_vector = None
    for arm in arms.T:  # Iterate over the transposed arms to get each arm as a column vector
        reward = get_real_reward(theta, arm)
        if reward > max_reward:
            max_reward = reward
            best_arm_vector = arm
    return best_arm_vector


def g_optimal(arm_matrix: np.ndarray, indexes: list):
    """

    :param arm_matrix: a matrix of current arm vectors
    :param indexes: indices of
    :return: a vector of probabilities (0,1) for each arm , using some optimization program
    """
    d, k = np.shape(arm_matrix)
    print(f"k={k},d={d}")
    num_vecs = len(indexes)
    print(f"arm matrix dims = {arm_matrix.shape}")
    print(f"indexes = {indexes}")
    # print(f"arm matrix = {arm_matrix}")
    def V(pi):
        #print(f"pi dims  = {pi.shape}")
        res = np.zeros((d,d))
        for i in indexes:
            res += np.asarray(pi[i] * (arm_matrix[:, i].reshape((d, 1)) @ (arm_matrix[:, i].reshape((d, 1)).T)))
        return res
    # V = lambda pi: sum(
    #     np.asarray([pi[i] * (arm_matrix[:, i].reshape((d, 1)) @ (arm_matrix[:, i].reshape((d, 1)).T))]) for i in
    #     indexes)  # finding the matrix V for a distribution
    pi_0 = (1 / num_vecs) * np.ones((num_vecs))  # initial guess to find pi

    def g(pi):
        """
        :param pi: a probability distribution
        :return: the best arm in given distribution using a_i^T V(pi)^-1 a as the mahalobanius norm
        """
        V_pi: np.ndarray = V(pi)
        # print(f"Det = {np.linalg.det(V_pi)}")
        # print(f"V_pi = \n {V_pi}")
        return np.max([(arm_matrix[:, i].reshape((d, 1))).T @ np.linalg.inv(V_pi).reshape((d, d)) @ (
            arm_matrix[:, i].reshape((d, 1))) for i in indexes])

    constraints = lambda v: np.sum(v) - 1  # sum of probabilities must equal 1
    constraints_dict = {'type': 'eq', 'fun': constraints}
    pi_0 = (1 / num_vecs) * np.ones((k))
    bounds = [(0, 1) if i in indexes else (0, 0) for i in range(k)]
    res = optimize.minimize(fun=g, bounds=bounds, constraints=constraints_dict, x0=pi_0)
    val = res['fun']
    pi = res['x']
    solver = res['message']
    return pi, solver


def simulate_fixed_budget(k=50, d=6, T=400, mean=0, sigma=1, num_trials=1):
    """

    :param k:  number of samples(arms)
    :param d: the dimension of each arm
    :param T: the budget for the simulation
    :param mean: the mean of the AWGN
    :param sigma: the standard deviation of the AWGN
    :param num_trials : number of trials to average the error
    :return:the optimal arm from the list of arms generated
    """
    plot_data = []
    original_arm_vectors, original_theta_star = generate_linear_bandit_instance(k, d)
    real_best_rewards =  np.asarray([original_theta_star.T @ original_arm_vectors[:,i] for i in range(k)]).reshape((1,k))
    plot_data.append({"r":0,"rewards":real_best_rewards,"indexes":list(range(k))})
    #print(f"Arm Vectors =\n {original_arm_vectors}")
    curr_arms = original_arm_vectors
    curr_indexes = list(range(k))
    logd = math.ceil(math.log2(d))
    print(f"logd={logd}")
    curr_dim = d
    new_dim = d
    m = (T - np.min([k, (d * (d + 1) / 2)]) - sum([d / (2 ** r) for r in range(1, logd)])) / logd
    real_best_reward = best_reward_vec(original_arm_vectors, original_theta_star)
    for r in range(1, 100):
        #check if new dim
        # print(curr_arms.shape)
        # new_dim = np.linalg.matrix_rank(curr_arms)
        # print(f"new dim ={new_dim}")
        # if (new_dim != curr_dim):
        #     (B, R) = np.linalg.qr(curr_arms, mode='reduced')
        #     print("SDSDSDSDS")
        #     print(f"k={k},d={d},B shape = {B.shape}")
        #     print(np.linalg.matrix_rank(B))
        if(len(curr_indexes) == 1):
            print("finished")
            print(f"winner index ={curr_indexes[0]}")
            break
        if r == 1:
            pi, solver = g_optimal(curr_arms, curr_indexes)
            #print(f"PI = {pi}r={r}")  # vector of probabilities
        elif r != 1:
            pi, solver = g_optimal(curr_arms, curr_indexes)
            #print(f"PI = {pi},r={r}")

        T_r_array = np.zeros((k))
        for i in curr_indexes:
            T_r_array[i] = np.ceil(pi[i] * m)
        Tr = np.sum(T_r_array)
        #print(f"T_rs = {T_r_array}")
        #print(f"Tr = {Tr}")
        curr_d = curr_arms.shape[0]
        #print(f"currarms[:,0]={curr_arms[:,0]}")
        V_r = np.zeros((curr_d,curr_d))
        print(V_r.shape)
        for i in curr_indexes:
            curr_vector = curr_arms[:,i].reshape((curr_d,1))
            V_r += (T_r_array[i] * (curr_vector@curr_vector.T))
        print(f"V_r ={V_r}")
        print(V_r.shape)
        newsum = np.zeros((curr_dim, 1))  # from 19 in alg
        for idx in curr_indexes:
            for j in range(int(T_r_array[idx])):
                X_t = get_reward(original_theta_star, original_arm_vectors[:,idx])
                #print(f" newsum shape = {newsum.shape},X_t = {X_t}")
                #print(f" curr_arms[:,idx].reshape((curr_dim, 1)) = {curr_arms[:,idx].reshape((curr_dim, 1)).shape}")
                newsum += curr_arms[:,idx].reshape((curr_dim, 1)) * X_t
        Theta_r = np.linalg.inv(V_r) @ newsum  # this is Theta_r as in phase 19
        expected_rewards = np.zeros(k)
        # print("LN 181")
        # print(f"expected_rewards shape = {expected_rewards.shape}")
        # print(f"Theta_r shape = {Theta_r.shape}")
        # print(f"Curr arms shape= {curr_arms.shape}")
        for idx in curr_indexes:
            expected_rewards[idx] = Theta_r.T@curr_arms[:,idx].reshape((curr_dim,1))  # saved all the expected rewards
        sorted_indexes = np.argsort(-expected_rewards)  # Sort in descending order
        # new_indexes = sorted_indexes[:len(curr_indexes) // 2]  # Select the top half with highest rewards
        if r == 1:
            new_indexes = sorted_indexes[:d]  # Select the top d/2 with highest rewards in the first iteration
        else:
            new_indexes = sorted_indexes[:len(curr_indexes) // 2]  # Select the top half with highest rewards from second iteration
        plot_data.append({"r":r,"indexes":new_indexes,"rewards":expected_rewards})
        curr_indexes = new_indexes
    return plot_data,original_theta_star,original_arm_vectors


def make_plots(plot_data:list):
    """

    :param plot_data: a list of r stages, each element is a dict with keys "r"(round),"indexes"(current indexes),"rewards" which are the expected rewards
    :return: make a matplotlib from each stage, where x values will be indexes,y values will be expected rewards, at stage 0 those are the real rewards with no noise
    """
    k = len(plot_data[0]['indexes'])
    num_rounds = len(plot_data)
    for stage in plot_data:
        round_number = stage['r']
        indexes = stage['indexes']
        rewards = stage['rewards'].flatten()
        reduced_rewards = [rewards[i] for i in indexes]

        # Create a histogram for each round
        plt.figure()
        print("indexes = ",indexes)
        print("rewards  = ",rewards)
        plt.bar(indexes, reduced_rewards, tick_label=indexes)
        plt.title(f'Round {round_number}')
        plt.xlabel('Index')
        plt.ylabel('Reward')
        plt.show()


if __name__ == "__main__":
    plot_data,original_theta_star,original_arm_vectors = simulate_fixed_budget()
    make_plots(plot_data)
