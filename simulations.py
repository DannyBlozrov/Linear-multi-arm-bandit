import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import random


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
    arm = np.reshape(arm, (armdim, 1))
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

    def V(pi):
        res = np.zeros((d, d))
        for i in indexes:
            res += np.asarray(pi[i] * (arm_matrix[:, i].reshape((d, 1)) @ (arm_matrix[:, i].reshape((d, 1)).T)))
        return res

    pi_0 = (1 / num_vecs) * np.ones((num_vecs))  # initial guess to find pi

    # def g(pi):
    #     """
    #     :param pi: a probability distribution
    #     :return: the best arm in given distribution using a_i^T V(pi)^-1 a as the mahalobanius norm
    #     """
    #     V_pi = V(pi)
    #     regularization_term = 1e-5 * np.eye(d)  # Regularization term
    #     V_pi += regularization_term
    #     return np.max([(arm_matrix[:, i].reshape((d, 1))).T @ np.linalg.inv(V_pi).reshape((d, d)) @ (arm_matrix[:, i].reshape((d, 1))) for i in indexes])

    def g(pi):
        """
        :param pi: a probability distribution
        :return: the best arm in given distribution using a_i^T V(pi)^-1 a as the mahalobanius norm
        """
        V_pi = V(pi)
    
        # Check if V_pi is singular or nearly singular
        try:
            np.linalg.inv(V_pi)
            # If no exception, the matrix is not singular
            return np.max([(arm_matrix[:, i].reshape((d, 1))).T @ np.linalg.inv(V_pi).reshape((d, d)) @ (arm_matrix[:, i].reshape((d, 1))) for i in indexes])
        except np.linalg.LinAlgError:
            # Matrix is singular; apply regularization
            regularization_term = 1e-5 * np.eye(d)
            V_pi += regularization_term
            return np.max([(arm_matrix[:, i].reshape((d, 1))).T @ np.linalg.inv(V_pi).reshape((d, d)) @ (arm_matrix[:, i].reshape((d, 1))) for i in indexes])

    constraints = lambda v: np.sum(v) - 1  # sum of probabilities must equal 1
    constraints_dict = {'type': 'eq', 'fun': constraints}
    pi_0 = (1 / num_vecs) * np.ones((k))
    bounds = [(0, 1) if i in indexes else (0, 0) for i in range(k)]
    res = optimize.minimize(fun=g, bounds=bounds, constraints=constraints_dict, x0=pi_0)
    val = res['fun']
    pi = res['x']
    solver = res['message']
    return pi, solver


def simulate_fixed_budget(k=200, d=5, T=400, mean=0, sigma=1, num_trials=1):
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
    histogram = np.zeros(k)
    original_arm_vectors, original_theta_star = generate_linear_bandit_instance(k, d)
    real_best_rewards = np.asarray([original_theta_star.T @ original_arm_vectors[:, i] for i in range(k)]).reshape((1, k))
    plot_data.append({"r": 0, "rewards": real_best_rewards, "indexes": list(range(k))})
    curr_arms = original_arm_vectors
    curr_indexes = list(range(k))
    logd = math.ceil(math.log2(d))
    print(f"logd={logd}")
    curr_dim = d
    new_dim = d
    m = (T - np.min([k, (d * (d + 1) / 2)]) - sum([d / (2 ** r) for r in range(1, logd)])) / logd
    real_best_reward = best_reward_vec(original_arm_vectors, original_theta_star)
    unused_indexes = []

    for r in range(1, 100):
        if len(curr_indexes) == 1:
            print("finished")
            print(f"winner index ={curr_indexes[0]}")
            break

        pi, solver = g_optimal(curr_arms, curr_indexes)
        T_r_array = np.zeros((k))
        for i in curr_indexes:
            T_r_array[i] = np.ceil(pi[i] * m)
        Tr = np.sum(T_r_array)
        curr_d = curr_arms.shape[0]
        V_r = np.zeros((curr_d, curr_d))

        for i in curr_indexes:
            curr_vector = curr_arms[:, i].reshape((curr_d, 1))
            V_r += (T_r_array[i] * (curr_vector @ curr_vector.T))

        # Add regularization term to V_r
        regularization_term = 1e-5 * np.eye(curr_d)
        V_r += regularization_term

        newsum = np.zeros((curr_dim, 1))
        for idx in curr_indexes:
            for j in range(int(T_r_array[idx])):

                X_t = get_reward(original_theta_star, original_arm_vectors[:, idx])
                newsum += curr_arms[:, idx].reshape((curr_dim, 1)) * X_t

        Theta_r = np.linalg.inv(V_r) @ newsum  # this is Theta_r as in phase 19
        expected_rewards = np.zeros(k)
        summed_rewards = np.zeros(k)

        if r ==1:
            for idx in curr_indexes:
                expected_rewards[idx] = Theta_r.T @ curr_arms[:, idx].reshape((curr_dim, 1))
                histogram[idx]+=1
        else:
            for idx in curr_indexes:
                random_vec = random.choice(unused_indexes)
                histogram[random_vec]+=1
                histogram[idx]+=1
                summed_rewards[idx] = Theta_r.T @ (curr_arms[:, idx].reshape(curr_dim, 1)+curr_arms[:, random_vec].reshape(curr_dim, 1))
                rand_reward = Theta_r.T @ curr_arms[:, random_vec].reshape(curr_dim, 1)
                expected_rewards[idx] = summed_rewards[idx]-rand_reward

        sorted_indexes = np.argsort(-expected_rewards)  # Sort in descending order

        if r == 1:
            new_indexes = sorted_indexes[:d]  # Select the top d/2 with highest rewards in the first iteration
        else:
            new_indexes = sorted_indexes[:len(curr_indexes) // 2]  # Select the top half with highest rewards from second iteration
        unused_indexes = [index for index in sorted_indexes if index not in new_indexes]
        #update histogram for trudy's freq_check

        plot_data.append({"r": r, "indexes": new_indexes, "rewards": expected_rewards})
        curr_indexes = new_indexes

    return plot_data, original_theta_star, original_arm_vectors, histogram


def make_plots(plot_data: list,histogram: list):
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
        print("indexes = ", indexes)
        print("rewards  = ", rewards)
        plt.bar(indexes, reduced_rewards, tick_label=indexes, width=0.4, color='b')
        plt.xlabel('Arm Index')
        plt.ylabel('Expected Reward')
        plt.title(f'Round {round_number}')
        plt.show()
    
    # Plot trudy's freq check
    plt.figure()
    plt.bar(range(len(histogram)),histogram)
    plt.xlabel('Arm Index')
    plt.ylabel('occurences')
    plt.title(f'Trudys histogram {round_number}')
    plt.show()


plot_data, original_theta_star, original_arm_vectors,histogram = simulate_fixed_budget(num_trials=1)
make_plots(plot_data,histogram)
