import math

from FW import FW_optimal
from functions import *


def simulate_vanilla(arms,theta,T,noise_params,threshold = 1e-5):
    """
    Simulates a bandit algorithm with a fixed budget, adjusting arm vectors when needed.
    
    :param arms:a matrix of size (d,k) which contains the arms as columns
    :param T: the budget for the simulation
    :param noise_params: a dict holding keys distribution == 'normal' 'mean':0,'std_dev':1 which are used for the noise generation
    :param threshold: the threshold for probabilities for which we decide 0 if smaller
    :return: the optimal arm from the list of arms generated
    """
    plot_data = []
    k = arms.shape[1]
    d = arms.shape[0]
    original_arm_vectors = arms
    original_theta_star = theta
    send_counter = 0
    real_best_rewards = np.asarray([original_theta_star.T @ original_arm_vectors[:, i] for i in range(k)]).reshape((1, k))
    plot_data.append({"r": 0, "rewards": real_best_rewards, "indexes": list(range(k)), "histogram": np.zeros((k))})
    
    curr_arms = original_arm_vectors
    curr_indexes = np.arange(k)
    logd = int(np.ceil(math.log2(d)))
    #print(f"logd={logd}")
    
    m = (T - np.min([k, (d * (d + 1) / 2)]) - sum([d / (2 ** r) for r in range(1, logd)])) / logd
    # print(f"m = {m}")
    real_best_reward = best_reward_vec(original_arm_vectors, original_theta_star)
    unused_indexes = []
    is_correct = 0
    print(f"theta = {theta}")
    d_r = d  # Start with initial dimension
    for r in range(1, int(logd)+1): #step 4
        print(f"curr indexes = {curr_indexes}")
        estimated_rewards = np.zeros(k)  # Estimated rewards (reset every round)
        histogram = np.zeros(k)  # Number of times each arm is pulled (reset every round)
        print(f"r = {r}")
        # Check if d_r == d_r-1 (dimension stays the same)
        #### RANK\DIMENSION Reduce
        d_r_prev = d_r
        d_r = np.linalg.matrix_rank(curr_arms)

        if d_r == d_r_prev:
            # Keep current arms as they are
                pass
        else:
            # Update arms with orthonormal basis matrix B_r
            # Find matrix B_r, whose columns form an orthonormal basis of the subspace
            _, B_r = np.linalg.qr(curr_arms[:, curr_indexes])
            # Update arm vectors by B_r^T * previous arm vectors
            for idx in curr_indexes:
                print(f"old dim = {curr_arms[:, idx].shape}")
                curr_arms[:, idx] = B_r.T @ curr_arms[:, idx]
                print(f"new dim = {curr_arms[:,idx].shape} ")

        #pi, solver = FW_optimal(curr_arms, curr_indexes) #setp 13/15
        pi, solver = g_optimal(curr_arms, curr_indexes)  # setp 13/15
        pi[pi < threshold] = 0
        pi = pi / np.sum(pi)  # Normalize to sum to 1
        #print(f"pi = {pi}")
        T_r_array = np.zeros(k,dtype=int)
        # Calculate T_r for each arm
        for i in curr_indexes:
            T_r_array[i] = int(np.ceil(pi[i] * m)) #step 17
        #print(f"T_r array = {T_r_array}")
        Tr = np.sum(T_r_array)
        V_r = np.zeros((d, d))
        acc = np.zeros(d)
        if r == 1:
            # print(f"curr arms = {curr_arms}")
            #doing step 19
            for idx in curr_indexes:
                outer_product = np.outer(curr_arms[:,idx],curr_arms[:,idx])
                outer_product = T_r_array[idx] * outer_product
                V_r += outer_product
                for w in range(T_r_array[idx]):
                    reward = get_reward(theta,curr_arms[:,idx],noise_params) #part of step 19
                    histogram[idx] += 1
                    send_counter += 1
                    acc += (curr_arms[:,idx] * reward) #the accumulator we use to find theta_hat
            # print(f"V_r = {V_r}")
            V_r_inverse = invert_matrix(V_r)
            theta_hat = V_r_inverse @ acc   #finishes step 19 of the alg
            print(f"theta hat = {theta_hat}")
            # print(f"theta hat = {theta_hat}")
            # print(f"curr indexes = {curr_indexes}")
            estimated_rewards = np.asarray([np.inner(curr_arms[:,i],theta_hat) for i in range(k)]) #step 20
            print(f"YYestimated rewards = {estimated_rewards}")
            curr_indexes = prune_indexes(estimated_rewards,curr_indexes,math.ceil(d / 2))  #step 21 done
            # print(f"new curr indexes = {curr_indexes}")
            unused_indexes = np.asarray([idx for idx in range(k) if idx not in curr_indexes]).flatten()
        else:

            # doing step 19
            for idx in curr_indexes:
                outer_product = np.outer(curr_arms[:, idx], curr_arms[:, idx])
                outer_product = T_r_array[idx] * outer_product
                V_r += outer_product
                for w in range(T_r_array[idx]):
                    reward = get_reward(theta, curr_arms[:, idx], noise_params)
                    histogram[idx] += 1
                    send_counter += 1
                    acc += (curr_arms[:, idx] * reward)

            V_r_inverse = invert_matrix(V_r)
            theta_hat = V_r_inverse @ acc  # finishes step 19 of the alg
            print(f"theta hat = {theta_hat}")
            # print(f"curr indexes = {curr_indexes}")
            estimated_rewards = np.asarray([np.inner(curr_arms[:, i], theta_hat) for i in range(k)])  # step 20
            print(f"XXestimated rewards = {estimated_rewards}")
            curr_indexes = prune_indexes(estimated_rewards, curr_indexes, math.ceil(d / (2**r)))  # step 21 done
            # print(f"new curr indexes = {curr_indexes}")
            unused_indexes = np.asarray([idx for idx in range(k) if idx not in curr_indexes]).flatten()
        plot_data.append({"r": r, "rewards": estimated_rewards, "indexes": curr_indexes, "histogram": histogram})
        if len(curr_indexes) == 1:
            # print(f"r = {r}")
            final_winner = curr_indexes[0]
            print(f"finnal winner = {final_winner},real best reward = {real_best_reward}")
            print(f"distance between winner and almost winner : {np.abs(get_real_reward(theta,curr_arms[:,final_winner])-get_real_reward(theta,curr_arms[:,real_best_reward]))}")
            if final_winner == real_best_reward:
                is_correct = 1
            break

    return plot_data, original_theta_star, original_arm_vectors,send_counter,is_correct 
