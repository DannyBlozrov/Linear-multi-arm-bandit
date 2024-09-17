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


def simulate_fixed_budget(k=50, d=5, T=400, num_trials=1,threshold = 1e-5):
    """
    :param k:  number of samples(arms)
    :param d: the dimension of each arm
    :param T: the budget for the simulation
    :param threshold: the threshold for probabilities for which we decide 0 if smaller
    :return:the optimal arm from the list of arms generated
    """
    plot_data = []
    original_arm_vectors, original_theta_star = generate_linear_bandit_instance(k, d)
    real_best_rewards = np.asarray([original_theta_star.T @ original_arm_vectors[:, i] for i in range(k)]).reshape(
        (1, k))
    plot_data.append({"r": 0, "rewards": real_best_rewards, "indexes": list(range(k)),"histogram":np.zeros((k))})
    curr_arms = original_arm_vectors
    curr_indexes = np.arange((k))
    logd = math.ceil(math.log2(d))
    print(f"logd={logd}")
    m = (T - np.min([k, (d * (d + 1) / 2)]) - sum([d / (2 ** r) for r in range(1, logd)])) / logd
    real_best_reward = best_reward_vec(original_arm_vectors, original_theta_star)
    print(f"real winner is : {real_best_reward}")
    unused_indexes = []


    for r in range(1, 100):
        if len(curr_indexes) == 1:
            print("finished")
            print(f"winner index ={curr_indexes[0]}")
            break
        estimated_rewards = np.zeros((k)) #the estimated rewards which will be 0 at every round
        histogram = np.zeros((k))     #the number of times arm i has been pulled ,will reset every round
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
        if r == 1:
            estimated_rewards = np.asarray([get_reward(original_theta_star,curr_arms[:,i]) for i in curr_indexes]).reshape((k))
            # print(f"estimated rewards shape = {estimated_rewards.shape}")
            # print(f"estimated rewards : {estimated_rewards}")
            top_indexes = np.argsort(estimated_rewards)[-d:]
            # print(f"top_indexes : {top_indexes}")
            curr_indexes =  np.asarray([idx for idx in top_indexes]).flatten()
            unused_indexes = np.asarray([idx for idx in range(k) if idx not in curr_indexes]).flatten()
            histogram += 1 #add 1 for every element, since we test them all
            # print(f"current indexes : {curr_indexes},unused indexes : {unused_indexes}")

        else:
            for idx in curr_indexes:
                if T_r_array[idx] != 0:
                    num_rows = int(T_r_array[idx])
                    histogram[idx] += num_rows
                    P = make_random_combinations_matrix(idx,num_rows,k,curr_arms,unused_indexes) # create the matrix P with T_r[i] rows and k columns
                    sums_columns = np.sum(P,axis=0)
                    for j in range(k):
                        histogram[j]  += sums_columns[j]
                    all_combinations_rewards = np.asarray([get_reward(original_theta_star,make_linear_combination(curr_arms,P[row])) for row in range(num_rows)]).reshape((num_rows))
                    # all combination_rewards[i] represents the reward from pulling the arm which is the summation of the arms with indexes from P[i] + noise

                    # print(f"all_combinations_rewards size = {all_combinations_rewards.shape}")
                    # print(f"all_combinations_rewards = {all_combinations_rewards}")
                    avg_reward = np.linalg.pinv(P) @ all_combinations_rewards #the OLS estimator for the rewards of arm0,arm1... we want arm[idx] because it has the most information
                    # print(f"avg_reward = {avg_reward}") #the average reward for each index 0,1,..k-1 , we are only interested in the idx
                    estimated_rewards[idx] = avg_reward[idx]
                    for idx2 in unused_indexes:
                        estimated_rewards[idx2] += avg_reward[idx2]

            for idx in unused_indexes:
                # print(f"estimated_rewards[{idx}]={estimated_rewards[idx]}")
                # print(f"histogram[{idx}] ={histogram[idx]}")
                estimated_rewards[idx] = estimated_rewards[idx]/histogram[idx]
        dummy_rounds = int(Tr/int(len(curr_indexes)))
        for pull in range(dummy_rounds):
            P = make_random_combinations_matrix(np.random.randint(k), 1, k, curr_arms,unused_indexes)  # create the matrix P with T_r[i] rows and k columns
            all_combinations_rewards = np.asarray([get_reward(original_theta_star, make_linear_combination(curr_arms, P[row])) for row in range(1)]).reshape((1))
            sums_columns = np.sum(P, axis=0)
            for j in range(k):
                histogram[j] += sums_columns[j]
        L = len(curr_indexes)
        print(f"curr indexes = {curr_indexes}")
        num_top_elements = int(np.ceil(L / 2))
        rewards_for_current_indexes = estimated_rewards[curr_indexes]
        top_indices_in_current = np.argsort(rewards_for_current_indexes)[-num_top_elements:]
        top_indexes = curr_indexes[top_indices_in_current]
        curr_indexes = top_indexes
        unused_indexes = np.asarray([i for i in range(k) if i not in curr_indexes]).flatten()
        print(f"Histogram at round {r} = {histogram}")
        plot_data.append({"r":r,"rewards":estimated_rewards,"indexes":curr_indexes,"histogram":histogram})








    return plot_data, original_theta_star, original_arm_vectors


def make_plots(plot_data: list):
    """
    :param plot_data: a list of r stages, each element is a dict with keys "r"(round),"indexes"(current indexes),"rewards" which are the expected rewards and "histogram
    :return: make a matplotlib from each stage, where x values will be indexes,y values will be expected rewards, at stage 0 those are the real rewards with no noise
    """
    k = len(plot_data[0]['indexes'])
    num_rounds = len(plot_data)
    cummulative_histogram = np.zeros((k))
    for stage in plot_data:
        round_number = stage['r']
        indexes = stage['indexes']
        rewards = stage['rewards'].flatten()
        reduced_rewards = [rewards[i] for i in indexes]
        cummulative_histogram = cummulative_histogram + stage['histogram']
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
        plt.bar(range(len(stage['histogram'])), stage['histogram'])
        plt.xlabel('Arm Index')
        plt.ylabel('occurences')
        plt.title(f'Trudys histogram at round {round_number}')
        plt.show()

    # Plot trudy's freq check
    plt.figure()
    plt.bar(range(len(cummulative_histogram)), cummulative_histogram)
    plt.xlabel('Arm Index')
    plt.ylabel('occurences')
    plt.title(f'Trudys Cummulative Histogram')
    plt.show()


plot_data, original_theta_star, original_arm_vectors = simulate_fixed_budget(num_trials=1)
make_plots(plot_data)
print(f"globalvar = {global_var}")
