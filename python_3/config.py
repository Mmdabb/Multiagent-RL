
# ==== Simulation Control Parameters ====
max_outer_iterations = 20
convergence_threshold = 1e-3

# ==== Agent Rollout Settings ====
mu_start = 0.3         # Initial μ (more exploration)
mu_end = 0.05          # Final μ (more greedy)
mu_decay_iter = 100    # Over how many iterations to decay μ

random_seed = 42
mu = 0.3
greedy = False

# ==== Flow Relaxation Settings (MSA) ====
use_msa = True

# ==== File Paths ====
data_path = "../data_sets/toy"
node_file = f"{data_path}/node.csv"
link_file = f"{data_path}/link.csv"
demand_file = f"{data_path}/demand.csv"
agent_result_file = f"{data_path}/agent_result.csv"
link_performance_file = f"{data_path}/link_performance.csv"
agent_link_flow_output = f"{data_path}/agent_implied_link_flow.csv"

# ==== Plotting Settings ====
save_plots = True
plot_output_folder = "../data_sets/3-corridor-acyclic/plots"
