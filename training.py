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

def population(q,t):
    x=q[0]
    y=q[1]
    a=q[2]
    dxdt=3*x*(1-x)-x*y-a*(1-exp(-5*x))
    dydt=-y+3*x*y
    dadt=0
    return [dxdt,dydt,dadt]

def visualize(model,epoch,device,rtol,folder):
    x0=1
    y0=0.01
    a_list=torch.tensor([0.62,0.64,0.66,0.68,0.7,0.75])

    t = np.linspace(0,100,1000)
    t_tensor=torch.from_numpy(t).float().to(device)

    fig_number = len(a_list)
    fig, axes = plt.subplots(2, fig_number,figsize=(5*fig_number, 10),sharey=True,sharex=True)

    #Create grid for vector field
    x = np.linspace(0, 1, 20)
    y = np.linspace(0, 2, 20)
    X, Y = np.meshgrid(x, y)
    
    for i, ax in enumerate(axes.flat):
        a0=a_list[i%fig_number]
        ax.set_xlim(0,1)
        ax.set_ylim(0,2)
        #True system
        if i<fig_number:
            ax.set_title(r'$\alpha=$'+f"{a0.item():.2f}")
            initial_contions=[x0,y0,a0]
            data = sc_int.odeint(population,initial_contions,t)        
            Xdot = 3*X*(1-X)-X*Y-a0.item()*(1-np.exp(-5*X))
            Ydot = -Y+3*X*Y
            ax.streamplot(X, Y, Xdot, Ydot,linewidth=1.5,color="silver")
            ax.plot(data[:,0],data[:,1], color="#503F9E",linewidth=1.5)            
            if i==0:
                ax.set(ylabel='y')
        #Learned system
        if i>=fig_number:
            ax.set(xlabel='x')
            initial_contions=torch.tensor([x0,y0,a0])

            grid_points = np.stack([X.ravel(), Y.ravel()], axis=-1)
            grid_points_tensor = torch.tensor(grid_points, dtype=torch.float32).to(device)
            
            with torch.no_grad():
                model.parameter_list=a0.unsqueeze(0)
                pred_x = odeint(model, initial_contions[:-1].float().unsqueeze(0),t_tensor,rtol=rtol)
                model.parameter_list=torch.full((grid_points_tensor.shape[0],),a0)
                derivatives = model(0, grid_points_tensor).cpu().numpy()
                Xdot = derivatives[:,0].reshape(X.shape)
                Ydot = derivatives[:, 1].reshape(X.shape)
                ax.streamplot(X, Y, Xdot, Ydot,linewidth=1.5,color="silver")  
                ax.plot(pred_x.cpu()[...,0],pred_x.cpu()[...,1],color="#503F9E",linewidth=1.5)                
    plt.tight_layout      
    plt.savefig(f"{folder}/epoch{epoch}.png")
    plt.close()

def create_grids(device):
    #Calculate derivatives on y-axis
    y_values = torch.arange(0.0, 2.1, 0.1,device=device)  
    a_values = torch.arange(0.58, 0.86, 0.02,device=device) 
    y_grid, a_grid = torch.meshgrid(y_values, a_values, indexing='ij')
    x = torch.zeros_like(y_grid.flatten(),device=device)
    y = y_grid.flatten()
    a = a_grid.flatten()

    yaxis_points = torch.stack([x, y, a], dim=1)
        
    #Calculate derivatives on x-axis
    x_values = torch.arange(0.0, 1.05, 0.05,device=device)  
    a_values = torch.arange(0.58, 0.86, 0.02,device=device) 
    x_grid, a_grid = torch.meshgrid(x_values, a_values, indexing='ij')
    y = torch.zeros_like(x_grid.flatten(),device=device)
    x = x_grid.flatten()
    a = a_grid.flatten()

    xaxis_points = torch.stack([x, y, a], dim=1)
    return xaxis_points, yaxis_points
    
def boundaryloss(model,device,xaxis_points,yaxis_points):

    model.parameter_list = xaxis_points[...,-1]
    derivatives_on_yaxis = model(0, yaxis_points[...,:-1])
    model.parameter_list = yaxis_points[...,-1]
    derivatives_on_xaxis = model(0, xaxis_points[...,:-1])

    x_derivatives = derivatives_on_yaxis[:,0]
    y_derivatives = derivatives_on_xaxis[:,1]

    x_true = torch.zeros_like(x_derivatives.flatten(),device=device)
    y_true = torch.zeros_like(y_derivatives.flatten(),device=device)    
    
    x_loss=torch.mean(abs(x_true - x_derivatives))
    y_loss=torch.mean(abs(y_true - y_derivatives))

    return x_loss+y_loss

if __name__ == "__main__":
    device = torch.device("cpu")
    
    ####Create argument parser####

    parser = argparse.ArgumentParser(description="Train model for vallidation")

    parser.add_argument('--epochs',         type=int, default=5000)
    parser.add_argument('--layers',         type=int, default=3)
    parser.add_argument('--depthlayers',    type=int,default=64)
    parser.add_argument('--trackloss',      type=bool, default=True)
    parser.add_argument('--learningrate',   type=float, default=0.0001)
    parser.add_argument('--batchlength',    type=int,default=20)
    parser.add_argument('--batchsize',      type=int,default=5)
    parser.add_argument('--regulation',     type=float,default=0.01)
    parser.add_argument('--rtol',           type=float,default=1e-5)
    parser.add_argument('--runname',        type=str,default="Test")
    parser.add_argument('--seed',           type=int,default="127")
    parser.add_argument('--dataset',        type=str,default="primary_dataset")
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
    folder=f"Runs/experiments/{runname}_{randomseed}"
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
                project="NB_experiments",
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
    dataname=f"Data/{dataset}.pth"

    data = torch.load(dataname)
    data_x = data["data_x"][:,:,:-1].float().to(device)
    data_a = data["data_x"][0,:,-1].float().to(device)
    data_t = data["data_t"].float().to(device)

    if validation!="none": #Only if we want to do validation

        dataname_val=f"Data/{validation}.pth"

        data_val = torch.load(dataname_val)

        val_dataname=f"Data/{validation}.pth"
        data_x_val = data_val["data_x"][:,:,:-1].float().to(device)
        data_a_val = data_val["data_x"][0,:,-1].float().to(device)
        data_t_val = data_val["data_t"].float().to(device)


    #Create dataframes from data
    dataframe = Dataframe(data_t, data_x, data_a, variables=2, drivers=1,batchlength=batchlength,batchsize=batchsize)
    if validation!="none": #Only if we want to do validation
        dataframe_validation = Dataframe(data_t_val, data_x_val, data_a_val,variables=2, drivers=1,batchlength=1,batchsize=1)
    
    #Create the model
    model=Neuralode(2, 1, parameter_list=data_a, hidden_layers=layers,depth_of_layers=depth).to(device)
    optimizer=torch.optim.Adam(model.parameters(), lr=learningrate)

    #Create grids on which to calculate physicsloss
    xaxis_points, yaxis_points = create_grids(device)
    
    for epoch in range(epochs):

        #Get batch list for this epoch
        batch_list,a_list,batch_t=dataframe.get_batch_list()

        for i,batch in enumerate(batch_list):
            
            optimizer.zero_grad()

            #Set bifurcation parameters for batch
            model.parameter_list=a_list[i]
            
            #Do forward passes through network+odesolver
            pred_x = odeint(model, batch[0,:,:].float(), batch_t.float(),rtol=rtol)
            
            #Calculate losses
            batch_loss = torch.mean(torch.abs(pred_x - batch)).to(device)
            physics_loss = boundaryloss(model,device,xaxis_points,yaxis_points)
            loss=batch_loss+labda*physics_loss

            if track:
                wandb.log({"batch_loss": loss})
        
            #Optimizer step
            loss.backward()
            optimizer.step()

        #Calculate training loss
        model.parameter_list=dataframe.data_a
        pred_x = odeint(model, dataframe.data_x[0,:,:].float(), dataframe.data_t.float(),rtol=rtol)
        train_loss = torch.mean(torch.abs(pred_x - dataframe.data_x))
        if track:
                wandb.log({"training loss": train_loss})
        
        if validation!="none":
            #Calculate validation loss
            model.parameter_list=dataframe_validation.data_a
            pred_x_test = odeint(model, dataframe_validation.data_x[0,:,:].float(), dataframe_validation.data_t.float(),rtol=rtol)
            val_loss = torch.mean(torch.abs(pred_x_test - dataframe_validation.data_x))
            if track:
                wandb.log({"validation loss": val_loss}) 

        if epoch%10==0:
            print(f"{runname} {randomseed}: {epoch}  training loss {train_loss.item()}")

        if epoch%100==0:
            #Save intermediate model
            file_name=f"{folder}/model.pth"
            torch.save(model.state_dict(), file_name)

        if epoch%500==0:
            #Visualize training process
            visualize(model,epoch,device,rtol,folder=folder)
             
    if track:                
        run.finish()

    file_name=f"{folder}/model.pth"
    torch.save(model.state_dict(), file_name)
    visualize(model,epochs,device,rtol,folder=folder) 
 
