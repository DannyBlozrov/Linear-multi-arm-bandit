import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from scipy.stats import entropy

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


def g_optimal(arm_matrix: np.ndarray, indexes: np.ndarray):
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


