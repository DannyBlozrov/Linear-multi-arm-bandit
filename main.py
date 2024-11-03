import numpy as np
from ls_linbai import ls_linbai
from od_linbai import od_linbai
from mod_linbai import mod_linbai
from multiprocessing import Pool
from functions import *
import importlib
import matplotlib.pyplot as plt


def run_simulation(config, algorithm_name):
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
    correct_counter = 0
    pulls = 0
    KLdiv = 0

    # Dynamically import the module containing the algorithm
    module = importlib.import_module(algorithm_name)
    algorithm_function = getattr(module, algorithm_name)

    arms, theta = generate_linear_bandit_instance(k, d, distribution_params, sorted=sorted)

    plot_data, original_theta_star, original_arm_vectors, total_appearances, is_correct = algorithm_function(
        arms, theta, config
    )

    correct_counter += is_correct
    pulls += total_appearances
    KLdiv += calculate_kl_divergence_with_uniform(plot_data)

    return {
        f'plot_data_{algorithm_name}': plot_data,
        f'{algorithm_name}_correct_counter': correct_counter,
        f'{algorithm_name}_pulls': pulls,
        f'{algorithm_name}_KLdiv': KLdiv,
    }


if __name__ == "__main__":
    config = load_config('config_paper.json')
    num_simulations = config.get('sim_num')
    algorithms = config.get('algorithms')
    error_probabilities_dict = {alg: [] for alg in algorithms}
    kl_divergence_dict = {alg: [] for alg in algorithms}

    k_values = list(range(25, 50, 3))
    for k in k_values:
        config['k'] = k
        results_per_algorithm = {}

        for algorithm in algorithms:
            successes = 0
            count = 0
            KLdiv = 0

            for _ in range(num_simulations):
                sim_results = run_simulation(config, algorithm)
                successes += sim_results[f'{algorithm}_correct_counter']
                count += sim_results[f'{algorithm}_pulls']
                KLdiv += sim_results[f'{algorithm}_KLdiv']

            error_probability = 1 - (successes / num_simulations)
            avg_KLdiv = KLdiv / num_simulations
            avg_count = count / num_simulations
            results_per_algorithm[algorithm] = {
                'error_prob': error_probability,
                'avg_armpull': avg_count,
                'avg_KLdiv': avg_KLdiv
            }
            error_probabilities_dict[algorithm].append(error_probability)
            kl_divergence_dict[algorithm].append(avg_KLdiv)

            print(f" for k = {k} and T=50 using {algorithm}: {results_per_algorithm[algorithm]}")

    # plot error prob
    for algorithm in algorithms:
        plt.plot(k_values, error_probabilities_dict[algorithm], marker='o', label=f'{algorithm} Error Probability')

    plt.xlabel(r'$K$ values')
    plt.ylabel('Error Probability')
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.title('Error Probability for Different Algorithms vs k for $T=50$')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

    #plot KLs
    for algorithm in algorithms:
        plt.plot(k_values, kl_divergence_dict[algorithm], marker='o', label=f'{algorithm} Average KL Divergence')

    plt.xlabel(r'$K$ values')
    plt.ylabel('Average KL Divergence')
    plt.title('Average KL Divergence for Different Algorithms vs k for $T=50$')
    plt.legend(loc='center right')
    plt.grid(True)
    plt.show()