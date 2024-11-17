import json
import numpy as np
import matplotlib.pyplot as plt
import importlib
from functions import load_config, generate_linear_bandit_instance, calculate_kl_divergence_with_uniform


def load_configs():
    """Load main and plot configurations from JSON files."""
    main_config = load_config('main_config.json')
    config_path = main_config.get('config_file')
    config = load_config(config_path)
    with open('plots_config.json', 'r') as f:
        plot_config = json.load(f)
    return config, plot_config.get("plot_settings", {})


def run_simulation(config, algorithm_name):
    """Run a single simulation using the specified algorithm."""
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
    # if  algorithm_name == "mod_linbai" and k > 20:
    #     data = plot_data[-1]['histogram']
    #     r = plot_data[-1]['r']
    #     print(f"k={k}")
    #     print(f"data = {data}")
    #     indices = range(k)
    #     plt.bar(indices, data, color='blue', edgecolor='black', alpha=0.7)
    #     plt.xticks(indices)
    #     plt.title('')
    #     plt.xlabel('Arm Index')
    #     plt.ylabel('Frequency')
    #     plt.show()
    correct_counter += is_correct
    pulls += total_appearances
    KLdiv += calculate_kl_divergence_with_uniform(plot_data)

    return {
        'correct_counter': correct_counter,
        'pulls': pulls,
        'KLdiv': KLdiv,
    }


def aggregate_results(config, algorithms, num_simulations, parameter_values, vary_parameter):
    """Run simulations across different parameter values (k or T) and aggregate results."""
    error_probabilities = {alg: [] for alg in algorithms}
    kl_divergence = {alg: [] for alg in algorithms}

    for value in parameter_values:
        config[vary_parameter] = value  # Set either k or T based on vary_parameter

        for algorithm in algorithms:
            successes, count, kl_div_sum = 0, 0, 0

            for _ in range(num_simulations):
                sim_results = run_simulation(config, algorithm)
                successes += sim_results['correct_counter']
                count += sim_results['pulls']
                kl_div_sum += sim_results['KLdiv']

            # Calculate error probability and average KL divergence
            error_probability = 1 - (successes / num_simulations)
            avg_KLdiv = kl_div_sum / num_simulations

            error_probabilities[algorithm].append(error_probability)
            kl_divergence[algorithm].append(avg_KLdiv)
            print(f"{vary_parameter}:{value}   ||| algorithm: {algorithm} |||  error probability:{error_probability} ||| avg KLdiv:{avg_KLdiv}")

    return error_probabilities, kl_divergence



def plot_results(k_values, error_probabilities, kl_divergence, plot_settings):
    """Plot the results using configurations from plot_settings."""
    # Plot Error Probability
    for algo_config in plot_settings.get("algorithms", []):
        algorithm = algo_config["name"]
        custom_label = algo_config.get("custom_label", algorithm)
        plt.plot(
            k_values,
            error_probabilities.get(algorithm, []),
            marker=plot_settings.get("marker", "o"),
            linestyle=plot_settings.get("line_style", "-"),
            label=custom_label
        )

    plt.xlabel(plot_settings.get("xlabel", "X-axis"))
    plt.ylabel(plot_settings.get("ylabel_error_prob", "Error Probability"))
    yticks = plot_settings.get("yticks", [0, 1.1, 0.1])
    plt.yticks(np.arange(yticks[0], yticks[1], yticks[2]))
    plt.legend(loc=plot_settings.get("legend_location_error_prob", "upper right"))
    plt.grid(plot_settings.get("grid", True))
    plt.title(plot_settings.get("plot_title_error_prob", "Error Probability Plot"))
    plt.show()

    # Plot KL Divergence
    for algo_config in plot_settings.get("algorithms", []):
        algorithm = algo_config["name"]
        custom_label = algo_config.get("custom_label", algorithm)
        plt.plot(
            k_values,
            kl_divergence.get(algorithm, []),
            marker=plot_settings.get("marker", "o"),
            linestyle=plot_settings.get("line_style", "-"),
            label=custom_label
        )

    plt.xlabel(plot_settings.get("xlabel", "X-axis"))
    plt.ylabel(plot_settings.get("ylabel_kl_div", "Average KL Divergence"))
    plt.legend(loc=plot_settings.get("legend_location_kl_div", "center right"))
    plt.grid(plot_settings.get("grid", True))
    plt.title(plot_settings.get("plot_title_kl_div", "KL Divergence Plot"))
    plt.show()



if __name__ == "__main__":
    config, plot_settings = load_configs()
    num_simulations = config.get('sim_num')
    algorithms = config.get('algorithms')
    vary_parameter = config.get("vary_parameter", "k")
    if vary_parameter == "k":
        values_range = config.get("k_values_range")
    elif vary_parameter == "T":
        values_range = config.get("T_values_range")
    else:
        raise ValueError("Invalid value for vary_parameter. Expected 'k' or 'T'.")

    parameter_values = list(range(values_range["start"], values_range["stop"], values_range["step"]))

    # Run simulations and aggregate results for the specified parameter
    error_probabilities, kl_divergence = aggregate_results(config, algorithms, num_simulations, parameter_values,
                                                           vary_parameter)

    # Plot the results, with x-axis label reflecting the varied parameter
    plot_results(parameter_values, error_probabilities, kl_divergence, plot_settings)