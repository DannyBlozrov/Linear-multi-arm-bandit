import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import random
global_var = 0

def arms_method(method:str,mean=0,variance=1):
    """

    :param method: a string such as 'normal' or 'uniform'
    :param mean: a float,mean
    :param variance: a float,variance
    :return: a lambda of (k,d) that will return an arm generating method [lambda]
    """
    if method == 'uniform':
        return lambda rows, cols: mean * np.random.rand(rows, cols)
    if method == 'normal':
        std_dev = np.sqrt(variance)
        return lambda rows,cols:np.random.normal(loc=mean,scale=std_dev,size=(rows,cols))

def generate_arm_vectors(k:int,d:int,method) -> np.ndarray:
    """

    :param k:number of vectors
    :param d: size of each vectors
    :param method : method to generate (higher order function), default is uniform [0,15]
    :return: a nparray of size (d,k)
    """
    arm_vectors = method(k,d)
    return arm_vectors
def generate_linear_bandit_instance(k, d):
    """
    :param K: number of samples of a_i..a_k
    :param d: the dimension of each sample
    :return:a matrix of arm vectors of size dxk, theta star of size dx1
    """
    method = arms_method('uniform',15)
    arm_vectors = generate_arm_vectors(k,d,method)
    theta = np.random.rand(d).reshape(d, 1)
    return arm_vectors.T,theta


def get_reward(theta, arm):
    # loc is the mean, scale is the variance
    armdim = arm.shape[0]
    arm = np.reshape(arm, (armdim, 1))
    awgn = np.random.normal(loc=0, scale=1, size=1)
    global global_var
    global_var += 1
    return theta.T @ arm + awgn


def get_real_reward(theta, arm):
    # Ensure arm is a column vector
    arm = arm.reshape(-1, 1)  # d x 1
    # Perform the matrix multiplication
    return (theta.T @ arm).item()  # Result is a scalar


def best_reward_vec(arms, theta):
    max_reward = -np.inf
    best_arm_index = None

    for i, arm in enumerate(arms.T):  # Iterate over the transposed arms to get each arm as a column vector
        reward = get_real_reward(theta, arm)
        if reward > max_reward:
            max_reward = reward
            best_arm_index = i

    return best_arm_index

def linear_combination(arms_set,indexes):
    """
    :param arms_set: a set of all possible arms
    :param indexes: the indexes u want to sum
    :return: a linear combination of the arms in indexes, IE if arms_set = [A1,A2,A3,A4] and indexes = [0,2] then return A1+A3
    """
    return np.sum(arms_set[:, indexes])
def solve_linear()
def g(pi, arm_matrix, indexes):
    """
    :param pi: a probability distribution
    :param arm_matrix: a matrix of arms
    :param indexes:a list of indexes
    :return: the best arm in given distribution using a_i^T V(pi)^-1 a as the mahalobanius norm
    """
    d = arm_matrix.shape[0]
    V_pi = np.zeros((d, d))
    for i in indexes:
        V_pi += np.asarray(pi[i] * (arm_matrix[:, i].reshape((d, 1)) @ (arm_matrix[:, i].reshape((d, 1)).T)))

    # Check if V_pi is singular or nearly singular
    try:
        np.linalg.inv(V_pi)
        # If no exception, the matrix is not singular
        return np.max([(arm_matrix[:, i].reshape((d, 1))).T @ np.linalg.inv(V_pi).reshape((d, d)) @ (
            arm_matrix[:, i].reshape((d, 1))) for i in indexes])
    except np.linalg.LinAlgError:
        # Matrix is singular; apply regularization
        regularization_term = 1e-5 * np.eye(d)
        V_pi += regularization_term
        return np.max([(arm_matrix[:, i].reshape((d, 1))).T @ np.linalg.inv(V_pi).reshape((d, d)) @ (
            arm_matrix[:, i].reshape((d, 1))) for i in indexes])


def g_optimal(arm_matrix: np.ndarray, indexes: list):
    """
    :param arm_matrix: a matrix of current arm vectors
    :param indexes: indices of
    :return: a vector of probabilities (0,1) for each arm , using some optimization program
    """
    print(arm_matrix.shape)
    d, k = np.shape(arm_matrix)
    print(f"k={k},d={d}")
    num_vecs = len(indexes)
    print(f"arm matrix dims = {arm_matrix.shape}")
    print(f"indexes = {indexes}")
    pi_0 = (1 / num_vecs) * np.ones((num_vecs))  # initial guess to find pi
    constraints = lambda v: np.sum(v) - 1  # sum of probabilities must equal 1
    constraints_dict = {'type': 'eq', 'fun': constraints}
    pi_0 = (1 / num_vecs) * np.ones((k))
    bounds = [(0, 1) if i in indexes else (0, 0) for i in range(k)]
    f = lambda pi: g(pi, arm_matrix, indexes)
    res = optimize.minimize(fun=f, bounds=bounds, constraints=constraints_dict, x0=pi_0)
    val = res['fun']
    pi = res['x']
    solver = res['message']
    return pi, solver


def simulate_fixed_budget(k=50, d=5, T=400, mean=0, sigma=0.5, num_trials=1,threshold = 1e-5):
    """
    :param k:  number of samples(arms)
    :param d: the dimension of each arm
    :param T: the budget for the simulation
    :param T: the threshold for probabilities for which we decide 0 if smaller
    :return:the optimal arm from the list of arms generated
    """
    plot_data = []
    histogram = np.zeros(k)
    original_arm_vectors, original_theta_star = generate_linear_bandit_instance(k, d)
    real_best_rewards = np.asarray([original_theta_star.T @ original_arm_vectors[:, i] for i in range(k)]).reshape(
        (1, k))
    plot_data.append({"r": 0, "rewards": real_best_rewards, "indexes": list(range(k))})
    curr_arms = original_arm_vectors
    curr_indexes = list(range(k))
    logd = math.ceil(math.log2(d))
    print(f"logd={logd}")
    m = (T - np.min([k, (d * (d + 1) / 2)]) - sum([d / (2 ** r) for r in range(1, logd)])) / logd
    real_best_reward = best_reward_vec(original_arm_vectors, original_theta_star)
    print(f"real winner is : {real_best_reward}")
    unused_indexes = []
    sent_vec_ind = []

    for r in range(1, 100):
        if len(curr_indexes) == 1:
            print("finished")
            print(f"winner index ={curr_indexes[0]}")
            break

        curr_rewards = []
        pi, solver = g_optimal(curr_arms, curr_indexes)
        pi[pi < threshold] = 0
        pi = pi/np.sum(pi)
        print(f"pi = {pi}")
        T_r_array = np.zeros((k))
        # calc T_r
        for i in curr_indexes:
            T_r_array[i] = np.ceil(pi[i] * m)
        Tr = np.sum(T_r_array)
        V_r = np.zeros((d, d))

        # getting the rewards
        newsum = np.zeros((d, 1))
        for idx in curr_indexes:
            for j in range(int(T_r_array[idx])):
                if r == 1:
                    histogram[idx] += 1
                    X_t = get_reward(original_theta_star, original_arm_vectors[:, idx])
                    temp_tuple = [idx, X_t]
                    curr_rewards.append(temp_tuple)
                    newsum += curr_arms[:, idx].reshape((d, 1)) * X_t
                # section added for correct count of histogram and choose different vecs for the same vec picks
                else:
                    random_vec = random.choice(unused_indexes)
                    temp_tuple = [idx, random_vec, 0]
                    histogram[random_vec] += 1
                    histogram[idx] += 1
                    X_t = get_reward(original_theta_star,
                                     (original_arm_vectors[:, idx] + original_arm_vectors[:, random_vec]))
                    temp_tuple[2] = X_t
                    curr_rewards.append(temp_tuple)
                    newsum += (curr_arms[:, idx].reshape((d, 1)) + curr_arms[:, random_vec].reshape(
                        (d, 1))) * X_t

        # use prev arms to calculate V_r
        if r == 1:
            for i in curr_indexes:
                curr_vector = curr_arms[:, i].reshape((d, 1))
                V_r += (T_r_array[i] * (curr_vector @ curr_vector.T))
        else:
            for i in sent_vec_ind:
                curr_vector = curr_arms[:, i[0]].reshape((d, 1)) + curr_arms[:, i[1]].reshape((d, 1))
                V_r += (T_r_array[i[0]] * (curr_vector @ curr_vector.T))

                # Add regularization term to V_r
        regularization_term = 1e-5 * np.eye(d)
        V_r += regularization_term
        est_rewards = np.zeros(k)
        Theta_r = np.linalg.inv(V_r) @ newsum  # this is Theta_r as in phase 19
        print(f"theta diff : {np.linalg.norm(original_theta_star - Theta_r)}")

        if r == 1:
            for idx in curr_indexes:
                est_rewards[idx] = Theta_r.T @ curr_arms[:, idx].reshape((d, 1))
        else:
            for idx in curr_indexes:
                # random_vec = random.choice(unused_indexes)
                # histogram[random_vec]+=1
                # histogram[idx]+=1
                est_rewards[idx] = Theta_r.T @ curr_arms[:, idx].reshape((d, 1))

        sorted_indexes = np.argsort(-est_rewards)  # Sort in descending order

        if r == 1:
            new_indexes = sorted_indexes[:d]  # Select the top d/2 with highest rewards in the first iteration
        else:
            new_indexes = sorted_indexes[
                          :len(curr_indexes) // 2]  # Select the top half with highest rewards from second iteration
        unused_indexes = [index for index in sorted_indexes if index not in new_indexes]
        # update histogram for trudy's freq_check

        plot_data.append({"r": r, "indexes": new_indexes, "rewards": est_rewards})
        curr_indexes = new_indexes

    return plot_data, original_theta_star, original_arm_vectors, histogram


def make_plots(plot_data: list, histogram: list):
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
        # print("indexes = ", indexes)
        # print("rewards  = ", rewards)
        plt.bar(indexes, reduced_rewards, tick_label=indexes, width=0.4, color='b')
        plt.xlabel('Arm Index')
        plt.ylabel('Expected Reward')
        plt.title(f'Round {round_number}')
        plt.show()

    # Plot trudy's freq check
    plt.figure()
    plt.bar(range(len(histogram)), histogram)
    plt.xlabel('Arm Index')
    plt.ylabel('occurences')
    plt.title(f'Trudys histogram {round_number}')
    plt.show()


plot_data, original_theta_star, original_arm_vectors, histogram = simulate_fixed_budget(num_trials=1)
make_plots(plot_data, histogram)
print(f"globalvar = {global_var}")
