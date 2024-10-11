import numpy as np
from bandit_mod import simulate_modified
from bandit_vanilla import simulate_vanilla
from multiprocessing import Pool
from functions import *





def run_simulation(config):
    k = config.get('k')
    d = config.get('d')
    T = config.get('T')
    seed = config.get('seed')
    seed_use = config.get('seed_use')
    if seed_use == "yes":
        np.random.seed(seed)
    else:
        np.random.seed()
    distribution_params = config.get('distribution_params')
    noise_params = config.get('noise_params')


    mod_correct_counter = 0
    vanilla_correct_counter = 0
    armpull_mod = 0
    armpull_vanilla = 0
    KLdiv_mod = 0
    KLdiv_vanilla = 0

    arms, theta = generate_linear_bandit_instance(k, d,distribution_params)

    # Run the simulation with user-defined parameters
    #plot_data_mod, original_theta_star, original_arm_vectors, total_appearances_mod, is_correct_mod = simulate_modified(arms, theta, T=T)
    plot_data_vanilla, original_theta_star, original_arm_vectors, total_appearances_vanilla, is_correct_vanilla = simulate_vanilla(
        arms, theta,config)

    #mod_correct_counter += is_correct_mod
    vanilla_correct_counter += is_correct_vanilla
    #armpull_mod += total_appearances_mod
    armpull_vanilla += total_appearances_vanilla
    #KLdiv_mod += calculate_kl_divergence_with_uniform(plot_data_mod)
    KLdiv_vanilla += calculate_kl_divergence_with_uniform(plot_data_vanilla)

    # Return the results from a single simulation
    return {
        'vanilla_correct_counter': vanilla_correct_counter,
        #'mod_correct_counter': mod_correct_counter,
        'armpull_vanilla': armpull_vanilla,
        #'armpull_mod':armpull_mod,
        'KLdiv_vanilla': KLdiv_vanilla,
        #'KLdiv_mod':KLdiv_mod
    }


# Main code to run simulations in parallel
if __name__ == "__main__":
    config = load_config('config_paper.json')
    num_simulations = config.get('sim_num')
    successes = 0
    for i in range(num_simulations):
        sim_results = run_simulation(config)
        succ = sim_results['vanilla_correct_counter']
        successes += succ
        for key in sim_results:
            print(key,sim_results[key])


    print(f"{successes}/{num_simulations} successes/simulations")



# Generate the plots from the simulation data
#make_plots(plot_data_mod)
#make_plots(plot_data_vanilla)

# Print the global variable (assuming it is defined in functions.py)
