import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import torch
import os

def lorenz(t, state, sigma, beta, r):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (r - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

def simulate_lorenz(sigma, beta, r, initial_state, t_span=(0, 40), dt=0.01):
    t_eval = np.arange(t_span[0], t_span[1], dt)
    sol = solve_ivp(
        lorenz, t_span, initial_state, args=(sigma, beta, r),
        t_eval=t_eval, rtol=1e-8, atol=1e-10
    )
    return sol.t, sol.y

def save_trajectory_plot(states, t, r, initial_state, save_dir, idx_traj):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(states[0], states[1], states[2])
    ax.set_title(f"Lorenz (r={r}, init={initial_state})")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim([-25, 25])
    ax.set_ylim([-20, 30])
    ax.set_zlim([0, 40])
    plt.tight_layout()
    fname = f"lorenz_traj{idx_traj}_r{r}_init{initial_state}.png"
    plt.savefig(os.path.join(save_dir, fname))
    plt.close(fig)
    print(f"Saved plot: {fname}")

def generate_lorenz_data(r_list, initial_conditions, t_span=(0, 40), dt=0.01, sigma=10.0, beta=8/3):
    save_dir = "lorenz_trajectories"
    os.makedirs(save_dir, exist_ok=True)
    num_r = len(r_list)
    num_init = len(initial_conditions)
    t_eval = np.arange(t_span[0], t_span[1], dt)
    num_timesteps = len(t_eval)
    num_trajectories = num_r * num_init
    data = torch.zeros((num_timesteps, num_trajectories, 3), dtype=torch.float32)

    idx = 0
    for i, r in enumerate(r_list):
        for j, init in enumerate(initial_conditions):
            t, states = simulate_lorenz(sigma, beta, r, init, t_span, dt)
            # states shape: (3, T) -> transpose to (T, 3)
            data[:, idx, :] = torch.tensor(states.T, dtype=torch.float32)
            save_trajectory_plot(states, t, r, init, save_dir, idx)
            idx += 1
    return data, t_eval

# Generate data for the lorenz system

r_list = [5,7.5,10,12.5,15,17.5,20,22.5, 25, 27.5]  # Example r values
initial_conditions = [
    [1.0, 1.0, 1.0],
    [-1.0, -1.0, 1.0]
]
data, t = generate_lorenz_data(r_list, initial_conditions, t_span=(0, 5), dt=0.01)

sampling_number = 5 # Downsample by taking every 20th timestep
x_data = data[::sampling_number, :, :] 
r_tensor = torch.tensor(r_list, dtype=torch.float32).repeat_interleave(len(initial_conditions))
t_data = torch.tensor(t[::sampling_number], dtype=torch.float32)

#Save x_data and r_tensor in a pth file
data_all = {"x_data": x_data, "r_data": r_tensor, "t_data": t_data}
torch.save(data_all, "lorenz_dataset_clean.pth")

print(x_data.shape)
#Normalize the data
mask = r_tensor <= 22.5

selected_data = x_data[:, mask.squeeze(), :]  # shape: (timesteps, selected_trajectories, 3)
selected_r = r_tensor[mask]       # shape: (selected_trajectories,)

# Compute mean and std for x, y, z over all selected trajectories and timesteps
xyz_mean = selected_data.reshape(-1, 3).mean(dim=0)
xyz_std = selected_data.reshape(-1, 3).std(dim=0)

# Compute mean and std for r
r_mean = selected_r.mean()
r_std = selected_r.std()

# Normalize all data
data_norm = (x_data - xyz_mean) / xyz_std
r_tensor_norm = (r_tensor - r_mean) / r_std

print(data_norm.shape)
# Save normalized data in the same format as loaded
data_all_normalized = {
    "x_data": data_norm,
    "r_data": r_tensor_norm,
    "t_data": t_data
}
torch.save(data_all_normalized, "lorenz_dataset_normalized.pth")

print("Normalization complete.")
print(f"x, y, z mean: {xyz_mean.tolist()}, std: {xyz_std.tolist()}")
print(f"r mean: {r_mean.item()}, std: {r_std.item()}")
# Save normalization statistics for later use
norm_stats = {
    "xyz_mean": xyz_mean,
    "xyz_std": xyz_std,
    "r_mean": r_mean,
    "r_std": r_std
}
torch.save(norm_stats, "lorenz_normalization_stats.pth")
print("Normalization statistics saved to 'lorenz_normalization_stats.pth'.")

# Choose r values and find their indices
plot_r_values = [5, 10, 15, 20, 25]
num_inits = 2  # [1,1,1] and [-1,-1,1]
r_indices = [i for i, r in enumerate(r_tensor.tolist()) if r in plot_r_values for _ in range(num_inits)]
# Since r_tensor is repeated for each initial condition, get indices for both
indices = []
for r in plot_r_values:
    idxs = (r_tensor == r).nonzero(as_tuple=True)[0].tolist()
    indices.extend(idxs)

fig, axs = plt.subplots(1, len(plot_r_values), figsize=(20, 4), subplot_kw={'projection': '3d'})
if len(plot_r_values) == 1:
    axs = [axs]

for i, r in enumerate(plot_r_values):
    # For each r, get the two corresponding trajectories (by initial condition)
    idx1 = indices[i * num_inits]      # [1,1,1]
    idx2 = indices[i * num_inits + 1]  # [-1,-1,1]
    traj1 = x_data[:, idx1, :]  # shape: (timesteps, 3)
    traj2 = x_data[:, idx2, :]
    ax = axs[i]
    ax.plot(traj1[:, 0], traj1[:, 1], traj1[:, 2], color='blue', label='[1,1,1]')
    ax.plot(traj2[:, 0], traj2[:, 1], traj2[:, 2], color='purple', label='[-1,-1,1]')
    ax.set_title(f"r={r}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim([-25, 25])
    ax.set_ylim([-20, 30])
    ax.set_zlim([0, 40])
    ax.legend()

plt.tight_layout()
plt.savefig("true.png")
plt.show()