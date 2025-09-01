import torch
from torchdiffeq import odeint
import argparse
import os
import numpy as np
from dataframe import Dataframe
from neuralode import Neuralode
import random
import matplotlib.pyplot as plt
import scipy.integrate as sc_int
from math import exp
import wandb
import warnings

if __name__ == "__main__":

    #Surpress warnings
    
    warnings.filterwarnings("ignore")

    device = torch.device("cpu")
    
    ####Create argument parser####

    parser = argparse.ArgumentParser(description="Train model for vallidation")

    parser.add_argument('--epochs',         type=int, default=7500)
    parser.add_argument('--layers',         type=int, default=2)
    parser.add_argument('--depthlayers',    type=int,default=32)
    parser.add_argument('--trackloss',      type=bool, default=False)
    parser.add_argument('--learningrate',   type=float, default=0.001)
    parser.add_argument('--batchlength',    type=int,default=10)
    parser.add_argument('--batchsize',      type=int,default=16)
    parser.add_argument('--regulation',     type=float,default=0.01)
    parser.add_argument('--rtol',           type=float,default=1e-5)
    parser.add_argument('--runname',        type=str,default="Test")
    parser.add_argument('--seed',           type=int,default="128")
    parser.add_argument('--dataset',        type=str,default="lorenz_dataset")
    parser.add_argument('--validation_set', type=str,default="none")

    args = parser.parse_args()

    # Access the arguments
    epochs          =args.epochs
    layers          =args.layers
    depth           =args.depthlayers
    learningrate    =args.learningrate
    batchsize       =args.batchsize
    batchlength     =args.batchlength
    track           =args.trackloss
    labda           =args.regulation
    rtol            =args.rtol
    runname         =args.runname
    randomseed      =args.seed
    dataset         =args.dataset
    validation      =args.validation_set

    #Set random seed
    random.seed(randomseed)
    
    #Create folder to save results 
    folder=f"runs/{runname}_{randomseed}"
    if os.path.isdir(folder):
        user_input = input(f"Folder '{folder}' already exists. Do you want to continue? (y/n): ").strip().lower()
        if user_input != "y":
            raise FileExistsError(f"Folder '{folder}' already exists and user chose not to continue.")
    else:
        os.makedirs(folder, exist_ok=False)
        print(f"Folder '{folder}' created.")


    #Track using WandBs
    if track:
        run=wandb.init(
            # set the wandb project where this run will be logged
                project="lorenz_experiments",
                name=f"{runname}_{randomseed}",
        
            # track hyperparameters and run metadata
            config={
            "epochs"        :epochs,          
            "layers"        :layers,         
            "layerdepth"    :depth,         
            "learningrate"  :learningrate,
            "batchsize"     :batchsize,         
            "batchlength"   :batchlength,
            "regulation"    :labda,
            "data set"      :dataset,
            "rtol"          :rtol,
            "seed"          :randomseed
            }
        ) 

    #Load the data
    dataname=f"Data/lorenz_dataset_normalized.pth"

    data = torch.load(dataname)
    data_x = (data["x_data"][:,:16,:] ).float().to(device)
    data_a = (data["r_data"][:16] ).float().to(device)
    data_t = data["t_data"].float().to(device)

    print(data_x.shape)

    if validation!="none":
        data_x_val = (data["x_data"][:,14:16,:] ).float().to(device)
        data_a_val = (data["r_data"][14:16] ).float().to(device)
        data_t_val = data["t_data"].float().to(device)

    #Load the normalization parameters
    norm_vals = torch.load("Data/lorenz_normalization_stats.pth")
    x_mean = norm_vals["xyz_mean"].to(device)
    x_std = norm_vals["xyz_std"].to(device)
    r_mean = norm_vals["r_mean"].to(device)
    r_std = norm_vals["r_std"].to(device)
    print("Loaded normalization values:")
    print(f"x_mean: {x_mean}, x_std: {x_std}, r_mean: {r_mean}, r_std: {r_std}")
    
    #Create dataframes from data
    dataframe = Dataframe(data_t, data_x, data_a, variables=3, drivers=1,batchlength=batchlength,batchsize=batchsize)
    if validation!="none":
        dataframe_validation = Dataframe(data_t_val, data_x_val, data_a_val, variables=3, drivers=1,batchlength=1,batchsize=1)
    
    #Create the model
    model=Neuralode(3, 1, parameter_list=data_a, hidden_layers=layers,depth_of_layers=depth).to(device)
    optimizer=torch.optim.Adam(model.parameters(), lr=learningrate)
    
    for epoch in range(epochs):

        #Get batch list for this epoch
        batch_list,a_list,batch_t=dataframe.get_batch_list()

        for i,batch in enumerate(batch_list):
            
            optimizer.zero_grad()

            #Set bifurcation parameters for batch
            model.parameter_list=a_list[i]
            
            #Do forward passes through network+odesolver
            pred_x = odeint(model, batch[0,:,:].float(), batch_t.float(), method='rk4', options={'step_size': 0.05}).to(device)
            
            #Calculate losses
            loss = torch.mean(torch.abs(pred_x - batch)).to(device)
        
            #Optimizer step
            loss.backward()
            optimizer.step()
        
        if track:
                wandb.log({"batch_loss": loss})

        #Calculate training loss
        with torch.no_grad():
            model.parameter_list=dataframe.data_a
            pred_x = odeint(model, dataframe.data_x[0,:,:].float(), dataframe.data_t.float(), method='rk4', options={'step_size': 0.05})
            train_loss = torch.mean(torch.abs(pred_x - dataframe.data_x))
            if track:
                wandb.log({"training loss": train_loss})

        #Calculate validation loss
        if validation!="none":
            with torch.no_grad():
                #Calculate validation loss
                model.parameter_list=dataframe_validation.data_a
                pred_x_test = odeint(model, dataframe_validation.data_x[0,:,:].float(), dataframe.data_t.float(), method='rk4', options={'step_size': 0.05})
                val_loss = torch.mean(torch.abs(pred_x_test - dataframe_validation.data_x))
                if track:
                    wandb.log({"validation loss": val_loss}) 

        if epoch%10==0:
            print(f"{runname} {randomseed}: {epoch}  training loss {train_loss.item()}")

        if epoch % 500 == 0:
            # Save intermediate model
            file_name = f"{folder}/model.pth"
            torch.save(model.state_dict(), file_name)

            # Plot model trajectories for [1,1,1] and r in [5,10,15,20,25]
            # Normalize r values and initial state using loaded normalization stats
            plot_r_values = ((torch.tensor([5, 10, 15, 20, 25], dtype=torch.float32) - r_mean) / r_std).to(device)
            initial_state = ((torch.tensor([[1.0, 1.0, 1.0]] * 5, dtype=torch.float32) - x_mean) / x_std).to(device)  # shape [5, 3]
            t_plot = torch.linspace(0, 20, 500).float().to(device)  # 200 steps from 0 to 5

            model.parameter_list = plot_r_values
            # Prepare both initial conditions
            init1 = torch.tensor([[1.0, 1.0, 1.0]] * 5, dtype=torch.float32)
            init2 = torch.tensor([[-1.0, -1.0, 1.0]] * 5, dtype=torch.float32)
            init1_norm = ((init1 - x_mean) / x_std).to(device)
            init2_norm = ((init2 - x_mean) / x_std).to(device)

            # Forward pass for both initial conditions
            pred_traj1 = odeint(model, init1_norm, t_plot, method='rk4', options={'step_size': 0.05}).detach().cpu().numpy()  # [T, 5, 3]
            pred_traj2 = odeint(model, init2_norm, t_plot, method='rk4', options={'step_size': 0.05}).detach().cpu().numpy()  # [T, 5, 3]

            # Denormalize
            pred_traj1_denorm = pred_traj1 * x_std.cpu().numpy() + x_mean.cpu().numpy()
            pred_traj2_denorm = pred_traj2 * x_std.cpu().numpy() + x_mean.cpu().numpy()

            fig, axs = plt.subplots(1, len(plot_r_values), figsize=(20, 4), subplot_kw={'projection': '3d'})
            if len(plot_r_values) == 1:
                axs = [axs]
            for idx, r_val in enumerate(plot_r_values.squeeze().tolist()):
                ax = axs[idx]
                ax.plot(pred_traj1_denorm[:, idx, 0], pred_traj1_denorm[:, idx, 1], pred_traj1_denorm[:, idx, 2], color='blue', label='[1,1,1]')
                ax.plot(pred_traj2_denorm[:, idx, 0], pred_traj2_denorm[:, idx, 1], pred_traj2_denorm[:, idx, 2], color='purple', label='[-1,-1,1]')
                denorm_r = r_val * r_std.item() + r_mean.item()
                ax.set_title(f"r={denorm_r:.2f}")
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_zlabel("Z")
                ax.set_xlim([-25, 25])
                ax.set_ylim([-20, 30])
                ax.set_zlim([0, 40])
                ax.legend()
            plot_path = os.path.join(folder, f"model_trajectories_epoch{epoch}.png")
            plt.savefig(plot_path)
            plt.close(fig)
      
    if track:                
        run.finish()

    file_name=f"{folder}/model.pth"
    torch.save(model.state_dict(), file_name)
 
