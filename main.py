import numpy as np
from bandit_mod import simulate_modified
from bandit_vanillam2 import simulate_vanilla
from functions import make_plots


# Prompt user for input values
k = int(input("Enter the number of arms (k) [default is 50]: ") or 50)
d = int(input("Enter the dimension of each arm (d) [default is 5]: ") or 5)
T = int(input("Enter the budget for the simulation (T) [default is 400]: ") or 400)
sim_num = int(input("Enter the number of simulations (T) [default is 100]: ") or 100)

mod_correct_counter = 0
vanilla_correct_counter = 0
armpull_mod = 0
armpull_vanilla = 0
for i in range(sim_num):
    # Run the simulation with user-defined parameters
    plot_data_mod, original_theta_star, original_arm_vectors,total_appearances_mod,is_correct_mod = simulate_modified(k=k, d=d, T=T)
    plot_data_vanilla, original_theta_star, original_arm_vectors,total_appearances_vanilla,is_correct_vanilla = simulate_vanilla(k=k, d=d, T=T, num_trials=1)
    mod_correct_counter+=is_correct_mod
    vanilla_correct_counter+=is_correct_vanilla
    armpull_mod+=total_appearances_mod
    armpull_vanilla+=total_appearances_vanilla

#average of armpulls in each simulation
avg_armpull_mod = armpull_mod / sim_num
avg_armpull_vanilla = armpull_vanilla / sim_num

#error probability in each simulation
error_prob_mod = 1 - (mod_correct_counter / sim_num)
error_prob_vanilla = 1 - (vanilla_correct_counter / sim_num)


# Print avg armpulls
print("avg armpulls (modified):", avg_armpull_mod)
print("avg armpulls (vanilla):", avg_armpull_vanilla)

# Print Avg error probability 
print("Avg error probability (modified):", error_prob_mod)
print("Avg error probability (vanilla):", error_prob_vanilla)


# Generate the plots from the simulation data
#make_plots(plot_data_mod)
#make_plots(plot_data_vanilla)

# Print the global variable (assuming it is defined in functions.py)
