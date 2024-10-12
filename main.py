import numpy as np
from bandit_mod import simulate_modified
from bandit_vanilla import simulate_vanilla
from multiprocessing import Pool
from functions import *
import matplotlib.pyplot as plt

def run_simulation(config):
    k = config.get('k')
    d = config.get('d')
    seed = config.get('seed')
    seed_use = config.get('seed_use')
    if seed_use == "True":
        np.random.seed(seed)
    else:
        np.random.seed()
    distribution_params = config.get('distribution_params')
    mod_correct_counter = 0
    vanilla_correct_counter = 0
    armpull_mod = 0
    armpull_vanilla = 0
    KLdiv_mod = 0
    KLdiv_vanilla = 0

    arms, theta = generate_linear_bandit_instance(k, d, distribution_params)

    # Run the vanilla simulation
    plot_data_vanilla, original_theta_star, original_arm_vectors, total_appearances_vanilla, is_correct_vanilla = simulate_vanilla(
        arms, theta, config)
    vanilla_correct_counter += is_correct_vanilla
    armpull_vanilla += total_appearances_vanilla
    KLdiv_vanilla += calculate_kl_divergence_with_uniform(plot_data_vanilla)
    #make_plots(plot_data_vanilla)

    # Return the results from a single simulation
    return {
        'vanilla_correct_counter': vanilla_correct_counter,
        'armpull_vanilla': armpull_vanilla,
        'KLdiv_vanilla': KLdiv_vanilla
    }


if __name__ == "__main__":
    config = load_config('config.json')
    num_simulations = config.get('sim_num')

    k_values = [25]
    error_probabilities = []

    # Loop over different values of k
    for k in k_values:
        config['k'] = k  # Update k in the config
        successes = 0
        num_simulations = config.get('sim_num')
        for i in range(num_simulations):  # 300 simulations for each value of k
            sim_results = run_simulation(config)
            succ = sim_results['vanilla_correct_counter']
            count = sim_results['armpull_vanilla']
            successes += succ
        error_probability = 1 - (successes / num_simulations)
        error_probabilities.append(error_probability)
        print(f"k={k}: {successes}/{num_simulations} successes, Error probability = {error_probability}")

    # Plot the results
    plt.plot(k_values, error_probabilities, marker='o')
    plt.xlabel(r'$K$ values')
    plt.ylabel('Error Probability')
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.title('Error Probability vs k for paper settings')
    plt.grid(True)
    plt.show()
