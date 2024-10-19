import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import random
from functions import *
from FW import FW_optimal




def ls_linbai(arms,theta,config):
    """
    :param arms:a matrix of arms of size (k,d)
    :param theta: the optimal hidden vector
    :param config: set up for noise, k ,d ,T and such.
    :return:the optimal arm from the list of arms generated
    """
    noise_params = config.get('noise_params')
    plot_data = []
    k = arms.shape[1]
    d = arms.shape[0]
    T = config.get('T')
    optimization_method = config.get('optimization')  # either "danny" or "FW"
    threshold = config.get('threshold')  # tolerance for matrix inversion regularization and for error in g optimal
    original_arm_vectors = arms
    original_theta_star = theta
    send_counter = 0
    real_best_rewards = np.asarray([original_theta_star.T @ original_arm_vectors[:, i] for i in range(k)]).reshape(
        (1, k))
    plot_data.append({"r": 0, "rewards": real_best_rewards, "indexes": list(range(k)),"histogram":np.zeros((k))})
    curr_arms = original_arm_vectors
    curr_indexes = np.arange((k))
    logd = math.ceil(math.log2(d))
    m = (T - np.min([k, (d * (d + 1) / 2)]) - sum([d / (2 ** r) for r in range(1, logd)])) / logd
    real_best_reward = best_reward_vec(original_arm_vectors, original_theta_star)
    unused_indexes = []
    is_correct = 0


    for r in range(1, int(logd)+1):
        estimated_rewards = np.zeros((k)) #the estimated rewards which will be 0 at every round
        histogram = np.zeros((k))     #the number of times arm i has been pulled ,will reset every round
        if optimization_method == "FW":
            pi, solver = FW_optimal(curr_arms, curr_indexes,threshold=threshold)
        else:
            pi, solver = g_optimal(curr_arms, curr_indexes, threshold=threshold)
        pi[pi < threshold] = 0
        pi = pi/np.sum(pi)
        T_r_array = np.zeros((k),dtype=int)
        for i in curr_indexes:
            T_r_array[i] = np.ceil(pi[i] * m)
        Tr = np.sum(T_r_array)
        V_r = np.zeros((d, d))
        if r == 1:
            estimated_rewards = np.asarray([get_reward(original_theta_star,curr_arms[:,i],noise_params) for i in curr_indexes]).reshape((k))
            send_counter += len(curr_indexes)
            curr_indexes = prune_indexes(estimated_rewards,curr_indexes,math.ceil(d))  #step 21 done
            unused_indexes = np.asarray([idx for idx in range(k) if idx not in curr_indexes]).flatten()
            histogram += 1 #add 1 for every element, since we test them all
            # print(f"current indexes : {curr_indexes},unused indexes : {unused_indexes}")

        else:
            for idx in curr_indexes:
                if T_r_array[idx] != 0:
                    num_rows = T_r_array[idx]
                    histogram[idx] += num_rows
                    P = make_random_combinations_matrix(idx,num_rows,k,curr_arms,unused_indexes) # create the matrix P with T_r[i] rows and k columns
                    sums_columns = np.sum(P,axis=0)
                    for j in range(k):
                        histogram[j]  += sums_columns[j]
                    all_combinations_rewards = np.asarray([get_reward(original_theta_star,make_linear_combination(curr_arms,P[row]),noise_params) for row in range(num_rows)]).reshape((num_rows))
                    send_counter += num_rows
                    # all combination_rewards[i] represents the reward from pulling the arm which is the summation of the arms with indexes from P[i] + noise

                    # print(f"all_combinations_rewards size = {all_combinations_rewards.shape}")
                    # print(f"all_combinations_rewards = {all_combinations_rewards}")
                    avg_reward = np.linalg.pinv(P) @ all_combinations_rewards #the OLS estimator for the rewards of arm0,arm1... we want arm[idx] because it has the most information
                    # print(f"avg_reward = {avg_reward}") #the average reward for each index 0,1,..k-1 , we are only interested in the idx
                    estimated_rewards[idx] = avg_reward[idx]
                    for idx2 in unused_indexes:
                        estimated_rewards[idx2] += avg_reward[idx2]

            for idx in unused_indexes:
                pass
                # print(f"estimated_rewards[{idx}]={estimated_rewards[idx]}")
                # print(f"histogram[{idx}] ={histogram[idx]}")
                #estimated_rewards[idx] = estimated_rewards[idx]/histogram[idx]
        #dummy_rounds = int(0.5*Tr/int(len(curr_indexes)))
        dummy_rounds = 0
        for pull in range(dummy_rounds):
            P = make_random_combinations_matrix(np.random.randint(k), 1, k, curr_arms,unused_indexes)  # create the matrix P with T_r[i] rows and k columns
            all_combinations_rewards = np.asarray([get_reward(original_theta_star, make_linear_combination(curr_arms, P[row]),noise_params) for row in range(1)]).reshape((1))
            send_counter+=1
            sums_columns = np.sum(P, axis=0)
            for j in range(k):
                histogram[j] += sums_columns[j]
        L = len(curr_indexes)
        # print(f"curr indexes = {curr_indexes}")
        num_top_elements = int(np.ceil(L / 2))
        curr_indexes = prune_indexes(estimated_rewards,curr_indexes,num_top_elements)
        unused_indexes = np.asarray([i for i in range(k) if i not in curr_indexes]).flatten()
        # print(f"Histogram at round {r} = {histogram}")
        plot_data.append({"r":r,"rewards":estimated_rewards,"indexes":curr_indexes,"histogram":histogram})
        if len(curr_indexes) == 1:
            # print("finished")
            # print(f"winner index ={curr_indexes[0]}")
            final_winner = curr_indexes[0]
            if (final_winner == real_best_reward):
                is_correct = 1
            break


    return plot_data, original_theta_star, original_arm_vectors,send_counter,is_correct


