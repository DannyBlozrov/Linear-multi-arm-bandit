import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from scipy.stats import entropy
import json
import warnings

def load_config(file_path):
    with open(file_path, 'r') as file:
        config = json.load(file)
    return config

def generate_linear_bandit_instance(k:int, d:int,distribution_params:dict):
    """
    :param k: number of samples of a_i..a_k
    :param d: the dimension of each sample
    :param distribution_params:dict : a dict with keys {method,mean,std_dev,low,high} where method, representing 'uniform' , 'normal' or 'paper'
    :return:a matrix of arm vectors of size dxk, theta star of size dx1
    """
    arms,theta = None,None
    if distribution_params['method'] == 'paper':
        arms = np.zeros((2,k))
        for i in range(1,k-1):
            phi_i = np.random.normal(0,0.09)
            arms[0,i] = np.cos((np.pi / 4) + phi_i)
            arms[1, i] = np.sin((np.pi / 4) + phi_i)
        arms[0,0] = 1
        arms[0,1] = 0
        arms[0,k-1] = np.cos((3*np.pi / 4))
        arms[1, k-1] = np.sin((3*np.pi / 4))
        theta = np.asarray(([1,0])).T

    if distribution_params['method'] == 'normal':
        arms = np.random.normal(0,1,size = (d,k))
        theta_params = distribution_params['theta']
        if theta_params['method'] == "normal":
            theta_mean = theta_params['mean']
            theta_stddev = theta_params['std_dev']
            theta = np.random.normal(theta_mean,theta_stddev,size=(d,1))
        if theta_params['method'] == "uniform":
            theta_low = theta_params['low']
            theta_high = theta_params['high']
            theta = theta_low + (theta_high - theta_low) *np.random.rand(d,1)

    if distribution_params['method'] == 'uniform':
        high = distribution_params['high']
        low = distribution_params['low']
        arms = low + (high - low) * np.random.rand(d, k)
        theta_params = distribution_params['theta']
        if theta_params['method'] == "normal":
            theta_mean = theta_params['mean']
            theta_stddev = theta_params['std_dev']
            theta = np.random.normal(theta_mean, theta_stddev, size=(d))
        if theta_params['method'] == "uniform":
            theta_low = theta_params['low']
            theta_high = theta_params['high']
            theta = theta_low + (theta_high - theta_low) * np.random.rand(d)
    return arms,theta


def get_reward(theta, arm,noise_params):
    """

    :param theta: a vector of size (d,1)
    :param arm: an arm, a vector of size (d,1)
    :param noise_params: params for the noise we create
    :return:  the sum inner product of <arm,theta>+noise
    """

    noise = 0
    if noise_params['distribution'] == 'normal':
        mean = noise_params['mean']
        std_dev = noise_params['std_dev']
        noise = np.random.normal(mean,std_dev)

    return np.inner(theta,arm) + noise

def prune_indexes(rewards:np.ndarray,curr_indexes:np.ndarray,num_elements:int = None):
    """
    :param rewards: a vector with rewards of size (k)
    :param curr_indexes: a vector with good indexes
    :param num_elements: an int that says how many elements to keep
    :return: a new vector that is the subset of curr_indexes with the num_elements highest rewards
    """
    if not num_elements:
        L = len(curr_indexes)
        num_elements = int(np.ceil(L / 2))
    rewards_for_current_indexes = rewards[curr_indexes]
    top_indices_in_current = np.argsort(rewards_for_current_indexes)[-num_elements:]
    top_indexes = curr_indexes[top_indices_in_current]
    return top_indexes
def get_real_reward(theta, arm):
    return np.inner(theta,arm)  # Result is a scalar

def invert_matrix(M,regularization = 1e-8):
    """
    :param M: a square matrix we try to invert, if its non invertible - we add a regularization term and then invert
    :return: the inverse of the matrix, or a close approximation
    """
    try:
        res = np.linalg.inv(M)
        return res
    except np.linalg.LinAlgError:
        n = M.shape[0]
        regularization_mat = regularization * np.eye(n)
        M = M + regularization_mat
        return invert_matrix(M)




def best_reward_vec(arms, theta):
    """

    :param arms:a matrix of size (d,k) of arms
    :param theta: a vector of size (d,1)
    :return:
    """
    inner_products = np.dot(theta, arms)
    # print(f"real rewards = {inner_products}")
    max_index = np.argmax(inner_products)
    return max_index

def make_random_combinations_matrix(idx,rows,cols,arms,unused_indexes):
    """
    :param idx: the index of current indexes we want to create a matrix for
    :param rows: how mayn rows will this matrix have
    :param cols: how many columns will the matrix have
    :param arms : which vectors
    :param unused_indexes: which indexes will be in the binary vectors
    :return:a matrix of size (rows,cols) where every row has a binary vector of (1,0) that are created randomly
    """
    # print(f"rows = {rows},cols = {cols}")
    P = np.zeros((rows,cols))
    for i in range(rows):
        P[i][idx] = 1
        sample_size = np.random.randint(0,len(unused_indexes)) #generate a number of indexes that will be 1
        chosen_indexes_vector = np.random.choice(unused_indexes,size=sample_size,replace=False) #returns a vector like [1,2,6,12] from the unused indexes
        P[i][chosen_indexes_vector] = 1

    # print(f"P = {P}")
    return P

def make_linear_combination(arms,index_vector):
    """
    :param arms: a matrix of all the arms of size (d,k)
    :param index_vector: a binary vector of size(k) whichi is like [0,1,0,1..] which represents taking all the odd index numbers
    :return: a vector which is the linear combination of them
    """
    selected_columns = arms[:, index_vector == 1]
    linear_combination = np.sum(selected_columns, axis=1)
    return linear_combination

def g(pi, arms, indexes):
    """
    :param pi: a probability distribution
    :param arms: a matrix of arms
    :param indexes:a list of indexes
    :return: the best arm in given distribution using a_i^T V(pi)^-1 a as the mahalobanius norm
    """
    d = arms.shape[0]
    V_pi = np.zeros((d, d))
    for idx in indexes:
        outer_product = np.outer(arms[:, idx], arms[:, idx])
        outer_product =pi[idx] * outer_product
        V_pi += outer_product
    V_inverse = invert_matrix(V_pi)
    all_norms = np.asarray([(arms[:,idx].T @ V_inverse) @ arms[:,idx]  for idx in indexes])
    return np.max(all_norms)


def g_optimal(arm_matrix: np.ndarray, indexes: np.ndarray, threshold, max_support=None):
    """
    :param arm_matrix: a matrix of current arm vectors
    :param indexes: indices of arms
    :param max_support: maximum number of non-zero elements allowed in pi
    :param tol: tolerance for small values to be considered as zero
    :return: a sparse vector of probabilities (0,1) for each arm, using optimization
    """
    d, k = np.shape(arm_matrix)
    max_support = max_support or (d * (d + 1)) // 2
    constraints = lambda v: np.sum(v) - 1
    constraints_dict = {'type': 'eq', 'fun': constraints}
    pi_0 = np.zeros(k)
    for i in indexes:
        pi_0[i] = 1 / len(indexes)
    bounds = [(0, 1) if i in indexes else (0, 0) for i in range(k)]
    f = lambda pi: g(pi, arm_matrix, indexes)
    res = optimize.minimize(fun=f, bounds=bounds, constraints=constraints_dict, x0=pi_0)
    pi = res['x']
    sorted_indices = np.argsort(-pi)  # Sort indices in descending order of pi
    support_indices = sorted_indices[:max_support]  # Select top max_support elements
    pi_thresholded = np.zeros_like(pi)
    pi_thresholded[support_indices] = pi[support_indices]
    pi_thresholded /= np.sum(pi_thresholded)
    active_indexes = [i for i in support_indices if pi_thresholded[i] > threshold]
    active_bounds = [(0, 1) if i in active_indexes else (0, 0) for i in range(k)]
    res_final = optimize.minimize(fun=f, bounds=active_bounds, constraints=constraints_dict, x0=pi_thresholded)
    pi_sparse = res_final['x']
    solver_message = res_final['message']

    return pi_sparse, solver_message

def apply_orthonormal_transformation(curr_arms, curr_indexes):
    subspace_arms = curr_arms[:, curr_indexes]
    Q, _ = np.linalg.qr(subspace_arms)
    transformed_arms = Q.T @ subspace_arms
    return transformed_arms

def make_plots(plot_data: list):
    """
    :param plot_data: a list of r stages, each element is a dict with keys "r"(round),"indexes"(current indexes),"rewards" which are the expected rewards and "histogram
    :return: make a matplotlib from each stage, where x values will be indexes,y values will be expected rewards, at stage 0 those are the real rewards with no noise
    """
    k = len(plot_data[0]['indexes'])
    cummulative_histogram = np.zeros((k))
    for stage in plot_data:
        round_number = stage['r']
        indexes = stage['indexes']
        rewards = stage['rewards'].flatten()
        reduced_rewards = [rewards[i] for i in indexes]
        cummulative_histogram += stage['histogram']
        # Create a histogram for each round
        plt.figure()
        plt.bar(indexes, reduced_rewards, tick_label=indexes, width=0.4, color='b')
        plt.xlabel('Arm Index')
        plt.ylabel('Expected Reward')
        plt.title(f'Round {round_number}')
        plt.show()
        # Plot histogram
        plt.figure()
        plt.bar(range(len(stage['histogram'])), stage['histogram'])
        plt.xlabel('Arm Index')
        plt.ylabel('Occurrences')
        plt.title(f'Histogram at round {round_number}')
        plt.show()

    # Plot cumulative histogram
    plt.figure()
    plt.bar(range(len(cummulative_histogram)), cummulative_histogram)
    plt.xlabel('Arm Index')
    plt.ylabel('Occurrences')
    plt.title('Cumulative Histogram')
    plt.show()


def calculate_kl_divergence_with_uniform(plot_data: list):
    """
    :param plot_data: a list of r stages, each element is a dict with keys "r"(round),"indexes"(current indexes),
                      "rewards" (expected rewards) and "histogram" (the number of times each arm was pulled).
    :return: The KL divergence between the cumulative histogram and a uniform distribution of the same size.
    """
    # Calculate cumulative histogram
    k = len(plot_data[0]['indexes'])  # Number of arms
    cumulative_histogram = np.zeros((k))

    for stage in plot_data:
        cumulative_histogram += stage['histogram']

    # Normalize cumulative histogram to create a probability distribution
    cumulative_histogram_prob = cumulative_histogram / np.sum(cumulative_histogram)

    # Create a uniform distribution of the same size
    uniform_distribution = np.ones(k) / k

    # Calculate KL divergence (using scipy's entropy function for KL divergence)
    kl_divergence = entropy(cumulative_histogram_prob, uniform_distribution)

    return kl_divergence


def find_index(index, transforms):
    """
    Find the original index that maps to the given final index through a series of transformations.

    :param index: The final index to trace back from.
    :param transforms: A list where each element is a dictionary with keys 'matrix' and 'mappings'.
                       'mappings' is a dictionary that maps old indexes to new indexes.
    :return: The original index that maps to the given final index.
    """
    # Start from the last transformation and trace back
    current_index = index
    for transform in reversed(transforms):
        # Get the current mapping
        mapping = transform['mappings']
        # Find the old index that maps to the current index
        for old_index, new_index in mapping.items():
            if new_index == current_index:
                current_index = old_index
                break
        else:
            # If we didn't find a mapping, the index cannot be traced back further
            return None

    # The final current_index will be the original index that maps to the given index
    return current_index




