import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ---------------------------
# Rossler system definition
# ---------------------------
def rossler(t, state, a, b, c):
    x, y, z = state
    dxdt = -y - z
    dydt = x + a * y
    dzdt = b + z * (x - c)
    return [dxdt, dydt, dzdt]

# ---------------------------
# Simulate Rossler
# ---------------------------
def simulate_rossler(a, b, c, initial_state, t_eval):
    t_span = (t_eval[0], t_eval[-1])
    sol = solve_ivp(
        rossler, t_span, initial_state, args=(a, b, c),
        t_eval=t_eval, rtol=1e-8, atol=1e-10
    )
    return sol.t, sol.y

# ---------------------------
# Save a single trajectory plot
# ---------------------------
def save_trajectory_plot(states, t, c, initial_state, save_dir, idx_traj):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(states[0], states[1], states[2])
    ax.set_title(f"Rössler (c={c}, init={initial_state})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xlim([-10, 15])
    ax.set_ylim([-15, 0])
    ax.set_zlim([0, 20])
    plt.tight_layout()
    fname = f"rossler_traj{idx_traj}_c{c}_init{initial_state}.png"
    plt.savefig(os.path.join(save_dir, fname))
    plt.close(fig)
    print(f"Saved plot: {fname}")

# ---------------------------
# Generate Rossler data
# ---------------------------
def generate_Rossler_data(c_list, initial_conditions, t_eval, a=0.2, b=0.2):
    save_dir = "Data/lorenz_trajectories"
    os.makedirs(save_dir, exist_ok=True)
    num_r = len(c_list)
    num_init = len(initial_conditions)
    num_timesteps = len(t_eval)
    num_trajectories = num_r * num_init
    data = torch.zeros((num_timesteps, num_trajectories, 3), dtype=torch.float32)

    idx = 0
    for i, c in enumerate(c_list):
        for j, init in enumerate(initial_conditions):
            t, states = simulate_rossler(a, b, c, init, t_eval)
            # states shape: (3, T) -> transpose to (T, 3)
            data[:, idx, :] = torch.tensor(states.T, dtype=torch.float32)
            save_trajectory_plot(states, t, c, init, save_dir, idx)
            idx += 1
    return data, t_eval

## We generate data for our test case
initial_state = [-4.0, -4.0, 0.0]  # Initial state for the Rössler system
c_list = np.arange(4.5, 5.6, 0.1)  
t_eval = np.linspace(0, 500, 20001) 

data, t_eval = generate_Rossler_data(c_list, [initial_state], t_eval, a=0.2, b=0.2)

data = data[19000:,:,:]
t_eval = t_eval[19000:]-t_eval[19000]


sampling_number = 5

data = data[::sampling_number, :, :]
data = data[:-1, :, :]
t_eval = t_eval[::sampling_number]
t_eval = torch.tensor(t_eval[:-1], dtype=torch.float32)

c_tensor = torch.tensor(c_list, dtype=torch.float32)
# Save raw data
data_all = {"x_data": data, "c_data":c_tensor, "t_data": t_eval}
torch.save(data_all, "rossler_dataset_not_normalized.pth")

# Directory to save the 3D scatter plots
save_dir = "Data/rossler_subsampled_plots"
os.makedirs(save_dir, exist_ok=True)

for idx in range(data.shape[1]):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(
        data[:, idx, 0].numpy(),
        data[:, idx, 1].numpy(),
        data[:, idx, 2].numpy(), cmap='viridis', s=10
    )
    ax.set_title(f"Rössler subsampled trajectory {idx}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.tight_layout()
    fname = f"rossler_subsampled_traj{idx}.png"
    plt.savefig(os.path.join(save_dir, fname))
    plt.close(fig)
    print(f"Saved subsampled plot: {fname}")

# ---------------------------
# Normalization (based on training data - validation)
# ---------------------------
selected_data = data[:, 1:, :] 
selected_c = c_tensor[1:]

# Compute mean and std for x, y, z
xyz_mean = selected_data.reshape(-1, 3).mean(dim=0)
xyz_std = selected_data.reshape(-1, 3).std(dim=0)

# Compute mean and std for r
c_mean = selected_c.mean()
c_std = selected_c.std()

# Normalize all data
data_norm = (data - xyz_mean) / xyz_std
c_tensor_norm = (c_tensor - c_mean) / c_std

print(data_norm.shape)

# Save normalized data
data_all_normalized = {
    "x_data": data_norm,
    "c_data": c_tensor_norm,
    "t_data": t_eval
}
torch.save(data_all_normalized, "rossler_dataset_normalized.pth")

print("Normalization complete.")
print(f"x, y, z mean: {xyz_mean.tolist()}, std: {xyz_std.tolist()}")
print(f"c mean: {c_mean.item()}, std: {c_std.item()}")

# Save normalization statistics for later use
norm_stats = {
    "xyz_mean": xyz_mean,
    "xyz_std": xyz_std,
    "c_mean": c_mean,
    "c_std": c_std
}
torch.save(norm_stats, "rossler_normalization_stats.pth")
print("Normalization statistics saved to 'rossler_normalization_stats.pth'.")

# Plot normalized data for each trajectory
save_dir_norm = "Data/rossler_normalized_subsampled_plots"
os.makedirs(save_dir_norm, exist_ok=True)

for idx in range(data_norm.shape[1]):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(
        data_norm[:, idx, 0].numpy(),
        data_norm[:, idx, 1].numpy(),
        data_norm[:, idx, 2].numpy(),
        c=t_eval.numpy(), cmap='viridis', s=10
    )
    ax.set_title(f"Rössler normalized trajectory {idx}")
    ax.set_xlabel("x (norm)")
    ax.set_ylabel("y (norm)")
    ax.set_zlabel("z (norm)")
    plt.tight_layout()
    fname = f"rossler_normalized_traj{idx}.png"
    plt.savefig(os.path.join(save_dir_norm, fname))
    plt.close(fig)
    print(f"Saved normalized plot: {fname}")

print(t_eval)