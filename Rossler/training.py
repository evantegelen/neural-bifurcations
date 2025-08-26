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

if __name__ == "__main__":
    device = torch.device("cpu")
    
    ####Create argument parser####

    parser = argparse.ArgumentParser(description="Train model for validation")

    parser.add_argument('--epochs',         type=int, default=100)
    parser.add_argument('--layers',         type=int, default=2)
    parser.add_argument('--depthlayers',    type=int,default=32)
    parser.add_argument('--trackloss',      type=bool, default=True)
    parser.add_argument('--learningrate',   type=float, default=0.001)
    parser.add_argument('--batchlength',    type=int,default=10)
    parser.add_argument('--batchsize',      type=int,default=16)
    parser.add_argument('--rtol',           type=float,default=1e-5)
    parser.add_argument('--runname',        type=str,default="Test")
    parser.add_argument('--seed',           type=int,default="127")
    parser.add_argument('--dataset',        type=str,default="rossler_dataset")
    parser.add_argument('--validation_set', type=str,default="yes")

    args = parser.parse_args()

    # Access the arguments
    epochs          =args.epochs
    layers          =args.layers
    depth           =args.depthlayers
    learningrate    =args.learningrate
    batchsize       =args.batchsize
    batchlength     =args.batchlength
    track           =args.trackloss
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
                project="rossler_experiments",
                name=f"{runname}_{randomseed}",
        
            # track hyperparameters and run metadata
            config={
            "epochs"        :epochs,          
            "layers"        :layers,         
            "layerdepth"    :depth,         
            "learningrate"  :learningrate,
            "batchsize"     :batchsize,         
            "batchlength"   :batchlength,
            "data set"      :dataset,
            "rtol"          :rtol,
            "seed"          :randomseed
            }
        ) 

    #Load the data
    dataname=f"rossler_data.pth"

    data = torch.load(dataname)
    data_x = (data["data"][:,0:8,:] * (1/10)).float().to(device)
    data_a = (data["c_list"][0:8] * (1/5)).float().to(device)
    data_t = data["sample_t"].float().to(device)

    if validation != "none":
        data_x_val = (data["data"][:,8:9,:] * (1/10)).float().to(device)
        data_a_val = (data["c_list"][8:9] * (1/5)).float().to(device)
        data_t_val = data["sample_t"].float().to(device)

    #Create dataframes from data
    dataframe = Dataframe(data_t, data_x, data_a, variables=3, drivers=1,batchlength=batchlength,batchsize=batchsize)
    if validation!="none": #Only if we want to do validation
        dataframe_validation = Dataframe(data_t_val, data_x_val, data_a_val,variables=2, drivers=1,batchlength=1,batchsize=1)
    
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
            pred_x = odeint(model, batch[0,:,:].float(), batch_t.float(), method='rk4', options={'step_size': 0.1}).to(device)
            
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
            pred_x = odeint(model, dataframe.data_x[0,:,:].float(), dataframe.data_t.float(), method='rk4', options={'step_size': 0.1})
            train_loss = torch.mean(torch.abs(pred_x - dataframe.data_x))
            if track:
                wandb.log({"training loss": train_loss})

        if validation!="none":
            with torch.no_grad():
                #Calculate validation loss
                model.parameter_list=dataframe_validation.data_a
                pred_x_test = odeint(model, dataframe_validation.data_x[0,:,:].float(), dataframe_validation.data_t.float(),rtol=rtol)
                val_loss = torch.mean(torch.abs(pred_x_test - dataframe_validation.data_x))
                if track:
                    wandb.log({"validation loss": val_loss}) 

        if epoch%10==0:
            print(f"{runname} {randomseed}: {epoch}  training loss {loss.item()}")

        if epoch % 100 == 0:
            # Save intermediate model
            file_name = f"{folder}/model.pth"
            torch.save(model.state_dict(), file_name)

        if epoch %500==0:
            #Visualize training process
            with torch.no_grad():
                plot_c_values = (torch.tensor([4.0, 6.0, 7.0], dtype=torch.float32)*1/10).to(device)
                initial_state = (torch.tensor([[-4.0, -4.0, 0.0]]*3, dtype=torch.float32)*1/10).to(device)  # shape [5, 3]
                t_plot = torch.linspace(0, 100, 200).float().to(device)  # 200 steps from 0 to 5

                model.parameter_list = plot_c_values

             # Forward pass for all r in one call
            pred_traj = odeint(model, initial_state, t_plot, method='rk4', options={'step_size': 0.01}).detach().cpu().numpy()  # [T, 5, 3]

            fig, axs = plt.subplots(1, len(plot_c_values), figsize=(20, 4), subplot_kw={'projection': '3d'})
            if len(plot_c_values) == 1:
                axs = [axs]
            for idx, r_val in enumerate(plot_c_values.squeeze().tolist()):
                ax = axs[idx]
                ax.plot(pred_traj[:, idx, 0], pred_traj[:, idx, 1], pred_traj[:, idx, 2])
                ax.set_title(f"r={r_val}")
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_zlabel("Z")
                ax.set_xlim([-10/10, 15/10])
                ax.set_ylim([-15/10, 0/10])
                ax.set_zlim([0/10, 20/10])
                #ax.scatter(data_x[:, :, 0].cpu().numpy(), data_x[:, :, 1].cpu().numpy(), data_x[:, :, 2].cpu().numpy(), color='red', s=1)
            plot_path = f"{folder}/model_trajectories_epoch{epoch}.png"
            plt.savefig(plot_path)
            plt.close(fig)
      
    if track:                
        run.finish()

    file_name=f"{folder}/model.pth"
    torch.save(model.state_dict(), file_name)
 
