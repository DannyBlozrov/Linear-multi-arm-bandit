import numpy as np
from bandit_mod import simulate_modified
from bandit_vanilla import simulate_vanilla
from multiprocessing import Pool
from functions import *



def run_simulation(args):
    k, d, T = args
    mod_correct_counter = 0
    vanilla_correct_counter = 0
    armpull_mod = 0
    armpull_vanilla = 0
    KLdiv_mod = 0
    KLdiv_vanilla = 0

    arms, theta = generate_linear_bandit_instance(k, d)

    # Run the simulation with user-defined parameters
    plot_data_mod, original_theta_star, original_arm_vectors, total_appearances_mod, is_correct_mod = simulate_modified(arms, theta, T=T)
    plot_data_vanilla, original_theta_star, original_arm_vectors, total_appearances_vanilla, is_correct_vanilla = simulate_vanilla(
        arms, theta, T=T)

    mod_correct_counter += is_correct_mod
    vanilla_correct_counter += is_correct_vanilla
    armpull_mod += total_appearances_mod
    armpull_vanilla += total_appearances_vanilla
    KLdiv_mod += calculate_kl_divergence_with_uniform(plot_data_mod)
    KLdiv_vanilla += calculate_kl_divergence_with_uniform(plot_data_vanilla)

    # Return the results from a single simulation
    return {
        'vanilla_correct_counter': vanilla_correct_counter,
        'mod_correct_counter': mod_correct_counter,
        'armpull_vanilla': armpull_vanilla,
        'armpull_mod':armpull_mod,
        'KLdiv_vanilla': KLdiv_vanilla,
        'KLdiv_mod':KLdiv_mod
    }


# Main code to run simulations in parallel
if __name__ == "__main__":
    # Prompt user for input values
    k = int(input("Enter the number of arms (k) [default is 50]: ") or 50)
    d = int(input("Enter the dimension of each arm (d) [default is 5]: ") or 5)
    T = int(input("Enter the budget for the simulation (T) [default is 400]: ") or 400)
    sim_num = int(input("Enter the number of simulations (Sim) [default is 100]: ") or 100)
    args_list = [(k, d, T) for _ in range(sim_num)]
    # Initialize multiprocessing pool
    with Pool() as pool:
        results = pool.map(run_simulation, args_list)

    # Aggregate results from all simulations
    vanilla_correct_counter = sum(result['vanilla_correct_counter'] for result in results)
    armpull_vanilla = sum(result['armpull_vanilla'] for result in results)
    KLdiv_vanilla = sum(result['KLdiv_vanilla'] for result in results)
    avg_armpull_vanilla = armpull_vanilla / sim_num
    error_prob_vanilla = 1 - (vanilla_correct_counter / sim_num)
    avg_KLdiv_vanilla = KLdiv_vanilla / sim_num
    print("avg armpulls (vanilla):", avg_armpull_vanilla)
    print("Avg error probability (vanilla):", error_prob_vanilla)
    print("Avg KL Divergence (vanilla):", avg_KLdiv_vanilla)


    mod_correct_counter = sum(result['mod_correct_counter'] for result in results)
    armpull_mod = sum(result['armpull_mod'] for result in results)
    KLdiv_mod = sum(result['KLdiv_mod'] for result in results)
    avg_armpull_mod = armpull_mod / sim_num
    error_prob_mod = 1 - (mod_correct_counter / sim_num)
    avg_KLdiv_mod = KLdiv_mod / sim_num
    print("avg armpulls (mod:", avg_armpull_mod)
    print("Avg error probability (mod):", error_prob_mod)
    print("Avg KL Divergence (mod):", avg_KLdiv_mod)



# Generate the plots from the simulation data
#make_plots(plot_data_mod)
#make_plots(plot_data_vanilla)

# Print the global variable (assuming it is defined in functions.py)
