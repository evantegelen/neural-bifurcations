import torch
import matplotlib.pyplot as plt
from neuralode import Neuralode
from torchdiffeq import odeint

# ---- User input: specify folder and model ----
folder = "runs/test11_121"
model_path = f"{folder}/model.pth"

# ---- Load model ----
device = torch.device("cpu")
# Adjust these parameters as needed to match your training
hidden_layers = 2
depth_of_layers = 32
parameter_list = torch.tensor([[4.0], [6.0], [7.0]], dtype=torch.float32) * 1/10

model = Neuralode(3, 1, parameter_list=parameter_list, hidden_layers=hidden_layers, depth_of_layers=depth_of_layers).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

with torch.no_grad():
    plot_c_values = (torch.tensor([4.0, 6.0, 7.0], dtype=torch.float32)*1/10).to(device)
    initial_state = (torch.tensor([[-4.0, -4.0, 0.0]]*3, dtype=torch.float32)*1/10).to(device)  # shape [3, 3]
    t_plot = torch.linspace(0, 100, 500).float().to(device)  # 500 steps from 0 to 100

    model.parameter_list = plot_c_values

    # Forward pass for all c in one call
    pred_traj = odeint(model, initial_state, t_plot, method='rk4', options={'step_size': 0.01}).detach().cpu().numpy()  # [T, 3, 3]

    fig, axs = plt.subplots(1, len(plot_c_values), figsize=(20, 4), subplot_kw={'projection': '3d'})
    if len(plot_c_values) == 1:
        axs = [axs]
    for idx, c_val in enumerate(plot_c_values.squeeze().tolist()):
        ax = axs[idx]
        ax.plot(pred_traj[:, idx, 0], pred_traj[:, idx, 1], pred_traj[:, idx, 2])
        ax.set_title(f"c={c_val}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_xlim([-1, 1.5])
        ax.set_ylim([-1.5, 0])
        ax.set_zlim([0, 2])
    plot_path = f"{folder}/model_trajectories_test.png"
    plt.savefig(plot_path)
    plt.close(fig)