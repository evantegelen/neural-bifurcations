import torch
from torchdiffeq import odeint
import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataframe import Dataframe
from neuralode import Neuralode
import random
import matplotlib.pyplot as plt
import scipy.integrate as sc_int
from math import exp

def population(q,t):
    x=q[0]
    y=q[1]
    a=q[2]
    dxdt=3*x*(1-x)-x*y-a*(1-exp(-5*x))
    dydt=-y+3*x*y
    dadt=0
    return [dxdt,dydt,dadt]

def visualize(model,epoch,device,rtol,folder):
    #Choose initial conditions
    x0=1
    y0=0.01
    #For which alpha to plot
    a_list=torch.tensor([0.62,0.64,0.66,0.68,0.7,0.75])

    #For which timespan to plot
    t = np.linspace(0,100,1000)
    t_tensor=torch.from_numpy(t).float().to(device)

    fig_number = len(a_list)
    fig, axes = plt.subplots(2, fig_number,figsize=(3.5*fig_number, 7),sharey=True,sharex=True)

    #Create grid for vector field
    x = np.linspace(0, 1, 20)
    y = np.linspace(0, 2, 20)
    X, Y = np.meshgrid(x, y)
    
    for i, ax in enumerate(axes.flat):
        a0=a_list[i%fig_number]
        
        ax.set_xlim(0,1)
        ax.set_ylim(0,1.5)
        #True system
        if i<fig_number:

            color="#060606"

            ax.set_title(r'$\alpha=$'+f"{a0.item():.2f}")
            #First compute trajectory
            initial_contions=[x0,y0,a0]
            data = sc_int.odeint(population,initial_contions,t)        
            #Create vector fields
            Xdot = 3*X*(1-X)-X*Y-a0.item()*(1-np.exp(-5*X))
            Ydot = -Y+3*X*Y
            
            #Plot the vector field
            ax.streamplot(X, Y, Xdot, Ydot,linewidth=1.5,color="silver")           
            #Plot example trajectory
            ax.plot(data[:,0],data[:,1], color=color,linewidth=1.5)  

            if i==0:
                ax.set(ylabel='y')
        
        #Learned system
        if i>=fig_number:
            color="#374B94"
            
            ax.set(xlabel='x')
            initial_contions=torch.tensor([x0,y0,a0])

            #Create grid to construct vector field
            grid_points = np.stack([X.ravel(), Y.ravel()], axis=-1)
            grid_points_tensor = torch.tensor(grid_points, dtype=torch.float32).to(device)
            
            with torch.no_grad():
                #First compute trajectory
                model.parameter_list=a0.unsqueeze(0)
                pred_x = odeint(model, initial_contions[:-1].float().unsqueeze(0),t_tensor,rtol=rtol)
                #create vector fields
                model.parameter_list=torch.full((grid_points_tensor.shape[0],),a0)
                derivatives = model(0, grid_points_tensor).cpu().numpy()
                Xdot = derivatives[:,0].reshape(X.shape)
                Ydot = derivatives[:, 1].reshape(X.shape)

                #Plot the vector field
                ax.streamplot(X, Y, Xdot, Ydot,linewidth=1.5,color="silver")  
                #Plot example trajectory
                ax.plot(pred_x.cpu()[...,0],pred_x.cpu()[...,1],color=color,linewidth=1.5)  
            
            if i==fig_number:
                ax.set(ylabel='y')              

    plt.tight_layout      
    plt.savefig(f"{folder}/vectorfield.png")
    plt.show()

if __name__ == "__main__":

    device = torch.device("cpu")
 
    model=Neuralode(2, 1, hidden_layers=3,depth_of_layers=64)
    model.load_state_dict(torch.load("Runs/experiments/primary_dataset/primary_dataset1_121/model.pth", weights_only=False))
    model.eval()

    visualize(model,0,device,1e-5,"Figures")