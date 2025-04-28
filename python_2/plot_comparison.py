import numpy as np
import matplotlib.pyplot as plt
import os
from config import *

# Load histories
multiagent_tt = np.loadtxt(os.path.join(data_path, "multiagent_system_travel_time.csv"), delimiter=",")
static_tt = np.loadtxt(os.path.join(data_path, "static_ue_system_travel_time.csv"), delimiter=",")

# Plot
plt.figure(figsize=(8,6))
plt.plot(multiagent_tt, label="Multi-Agent Stochastic Rollout", color="blue", linestyle="--")
plt.plot(static_tt, label="Static UE (MSA TAP)", color="red", linestyle="-")
plt.xlabel("Iteration")
plt.ylabel("Total System Travel Time (veh-minutes)")
plt.title("Comparison of UE vs Stochastic Multi-Agent Rollout")
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(data_path, "comparison_plot.png"))

plt.show()
