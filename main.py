import numpy as np
from ls_linbai import ls_linbai
from od_linbai import od_linbai
from mod_linbai import mod_linbai
from multiprocessing import Pool
from functions import *
import matplotlib.pyplot as plt

def run_simulation(config):
    k = config.get('k')
    d = config.get('d')
    sorted = config.get('sorted')
    seed = config.get('seed')
    seed_use = config.get('seed_use')
    if seed_use == "True":
        np.random.seed(seed)
    else:
        np.random.seed()
    distribution_params = config.get('distribution_params')
    od_linbai_correct_counter = 0
    od_linbai_pulls = 0
    od_linbai_KLdiv = 0

    arms, theta = generate_linear_bandit_instance(k, d, distribution_params,sorted = sorted)

    # Run the vanilla simulation
    plot_data_od_linbai, original_theta_star, original_arm_vectors, total_appearances_od_linbai, is_correct_od_linbai= od_linbai(
        arms, theta, config)
    od_linbai_correct_counter += is_correct_od_linbai
    od_linbai_pulls += total_appearances_od_linbai
    od_linbai_KLdiv += calculate_kl_divergence_with_uniform(plot_data_od_linbai)

    return {
        'plot_data_od_linbai':plot_data_od_linbai,
        'od_linbai_correct_counter': od_linbai_correct_counter,
        'od_linbai_pulls': od_linbai_pulls,
        'od_linbai_KLdiv': od_linbai_KLdiv,
    }


if __name__ == "__main__":
    config = load_config('config_paper.json')
    num_simulations = config.get('sim_num')
    error_probabilities = []
    k_values = list([k for k in range(25,50,3)])
    for k in k_values:
        config['k']=k
        successes_od_linbai = 0
        count_od_linbai = 0
        KLdiv_od_linbai = 0
        for i in range(num_simulations):  # 300 simulations for each value of k
            sim_results = run_simulation(config)
            successes_od_linbai += sim_results['od_linbai_correct_counter']
            count_od_linbai += sim_results['od_linbai_pulls']
            KLdiv_od_linbai += sim_results['od_linbai_KLdiv']
            # if i == 1 :
            #     make_plots(sim_results['plot_data_vanilla'])
        error_probability_vanilla = 1 - (successes_od_linbai / num_simulations)
        KLdiv_vanilla = KLdiv_od_linbai/num_simulations
        count_vanilla = count_od_linbai / num_simulations
        output_file_path = config['output_file']
        results = {'error_prob_vanilla':error_probability_vanilla,'armpull_vanilla':count_vanilla,'KLdiv_vanilla':KLdiv_vanilla}
        print(f" for k = {k} and T=50: {results}")
        error_probabilities.append(error_probability_vanilla)
    plt.plot(k_values, error_probabilities, marker='o')
    plt.xlabel(r'$K$ values')
    plt.ylabel('Error Probability')
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.title('Error Probability using OD-LINBAI vs k for $T=50$')
    plt.grid(True)
    plt.show()
