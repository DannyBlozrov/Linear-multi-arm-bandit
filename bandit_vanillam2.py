import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import random
from functions import (
    arms_method,
    generate_arm_vectors,
    generate_linear_bandit_instance,
    get_reward,
    get_real_reward,
    best_reward_vec,
    make_random_combinations_matrix,
    make_linear_combination,
    g,
    g_optimal,
    apply_orthonormal_transformation,
)
global_var_vanilla = 0
global_var__succ_vanilla = 0

def simulate_vanilla(k=50, d=5, T=400, num_trials=1, threshold=1e-5):
    """
    Simulates a bandit algorithm with a fixed budget, adjusting arm vectors when needed.
    
    :param k:  number of arms
    :param d: the dimension of each arm
    :param T: the budget for the simulation
    :param threshold: the threshold for probabilities for which we decide 0 if smaller
    :return: the optimal arm from the list of arms generated
    """
    plot_data = []
    send_counter = 0
    original_arm_vectors, original_theta_star = generate_linear_bandit_instance(k, d)
    real_best_rewards = np.asarray([original_theta_star.T @ original_arm_vectors[:, i] for i in range(k)]).reshape((1, k))
    plot_data.append({"r": 0, "rewards": real_best_rewards, "indexes": list(range(k)), "histogram": np.zeros((k))})
    
    curr_arms = original_arm_vectors
    curr_indexes = np.arange(k)
    logd = math.ceil(math.log2(d))
    print(f"logd={logd}")
    
    m = (T - np.min([k, (d * (d + 1) / 2)]) - sum([d / (2 ** r) for r in range(1, logd)])) / logd
    real_best_reward = best_reward_vec(original_arm_vectors, original_theta_star)
    print(f"real winner is : {real_best_reward}")
    unused_indexes = []

    d_r = d  # Start with initial dimension
    for r in range(1, 100):
        if len(curr_indexes) == 1:
            print("Finished")
            final_winner = curr_indexes[0]
            print(f"Winner index = {curr_indexes[0]}")
            break
        
        estimated_rewards = np.zeros(k)  # Estimated rewards (reset every round)
        histogram = np.zeros(k)  # Number of times each arm is pulled (reset every round)
        
        # Check if d_r == d_r-1 (dimension stays the same)
        if r > 1:
            d_r_prev = d_r
            d_r = curr_arms.shape[0]  # Dimension at round r
            
            if d_r == d_r_prev:
                # Keep current arms as they are
                for idx in curr_indexes:
                    estimated_rewards[idx] += get_reward(original_theta_star, curr_arms[:, idx])
                    send_counter+=1
            else:
                # Update arms with orthonormal basis matrix B_r
                # Find matrix B_r, whose columns form an orthonormal basis of the subspace
                _, B_r = np.linalg.qr(curr_arms[:, curr_indexes])
                
                # Update arm vectors by B_r^T * previous arm vectors
                for idx in curr_indexes:
                    curr_arms[:, idx] = B_r.T @ curr_arms[:, idx]
        
        pi, solver = g_optimal(curr_arms, curr_indexes)
        pi[pi < threshold] = 0
        pi = pi / np.sum(pi)  # Normalize to sum to 1
        
        print(f"pi = {pi}")
        T_r_array = np.zeros(k)
        
        # Calculate T_r for each arm
        for i in curr_indexes:
            T_r_array[i] = np.ceil(pi[i] * m)
        Tr = np.sum(T_r_array)
        
        V_r = np.zeros((d, d))
        if r == 1:
            estimated_rewards = np.asarray([get_reward(original_theta_star, curr_arms[:, i]) for i in curr_indexes]).reshape((k))
            send_counter+=len(curr_indexes)
            top_indexes = np.argsort(estimated_rewards)[-d:]
            curr_indexes = np.asarray([idx for idx in top_indexes]).flatten()
            unused_indexes = np.asarray([idx for idx in range(k) if idx not in curr_indexes]).flatten()
            histogram += 1  # Add 1 for every element, since we test them all
        else:
            for idx in curr_indexes:
                if T_r_array[idx] != 0:
                    num_rows = int(T_r_array[idx])
                    histogram[idx] += num_rows
                    for i in range(num_rows):
                        histogram[idx] += 1
                        reward = get_reward(original_theta_star, curr_arms[:, idx])
                        send_counter+=1
                        estimated_rewards[idx] += reward

            for idx in unused_indexes:
                estimated_rewards[idx] = estimated_rewards[idx] / histogram[idx]
        
        L = len(curr_indexes)
        num_top_elements = int(np.ceil(L / 2))
        rewards_for_current_indexes = estimated_rewards[curr_indexes]
        top_indices_in_current = np.argsort(rewards_for_current_indexes)[-num_top_elements:]
        top_indexes = curr_indexes[top_indices_in_current]
        curr_indexes = top_indexes
        unused_indexes = np.asarray([i for i in range(k) if i not in curr_indexes]).flatten()
        print(f"Histogram at round {r} = {histogram}")
        plot_data.append({"r": r, "rewards": estimated_rewards, "indexes": curr_indexes, "histogram": histogram})
    is_correct = 0
    print(f"final winner {r} = real best reward: {real_best_reward}")
    if final_winner == real_best_reward:
        is_correct = 1
    
    return plot_data, original_theta_star, original_arm_vectors,send_counter,is_correct