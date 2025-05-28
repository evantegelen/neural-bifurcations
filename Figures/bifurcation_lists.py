import matplotlib.pyplot as plt
import scipy.integrate as sc_int
import numpy as np
import warnings
import torch
from math import exp
warnings.simplefilter('always', UserWarning)
from torchdiffeq import odeint
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from neuralode import Neuralode

"""
This code can be used to generate the points necessary to generate the bifurcation diagrams.
The idea is to generate trajectorie data for different initial conditions and extrapolate the convergence behavior.
"""

#Which models to load
list_modelnames = [
    "Runs/experiments/limited_data/exp2_increase_alpha1_121/model.pth",
    "Runs/experiments/limited_data/exp2_increase_alpha2_232/model.pth",
    "Runs/experiments/limited_data/exp2_increase_alpha3_343/model.pth",
    "Runs/experiments/limited_data/exp2_increase_alpha4_454/model.pth",
    "Runs/experiments/limited_data/exp2_increase_alpha5_565/model.pth",
]

#How to save each model
list_filenames = [
    "Figures/bifurcation_points/ex2_increase_1.pth",
    "Figures/bifurcation_points/ex2_increase_2.pth",
    "Figures/bifurcation_points/ex2_increase_3.pth",
    "Figures/bifurcation_points/ex2_increase_4.pth",
    "Figures/bifurcation_points/ex2_increase_5.pth"
]

for _,modelname in enumerate(list_modelnames):

    #modelname="Runs/experiments/primary_dataset/primary_dataset1_121/model.pth"
    filename=list_filenames[_]

    visual=True

    #Load the model parameters
    model=Neuralode(2, 1, hidden_layers=3,depth_of_layers=64)
    model.load_state_dict(torch.load(modelname, weights_only=False))
    model.eval()

    #Time-span on which to calculate trajectories
    t = torch.from_numpy(np.linspace(0,2000,2000)).float()
    #Number of alpha samples to use
    N_a=500

    #Generates alpha list
    a_list= torch.tensor(np.linspace(0.6,0.75,N_a)).float()

    #Initial condition 1
    ic_list1 = torch.tensor([1.0, 0.01]).repeat(N_a, 1).float()
    #Initial condition 1
    ic_list2 = torch.tensor([0, 0]).repeat(N_a, 1).float()

    range_l=500 #Last ... timepoints to select min and max from.

    with torch.no_grad():
        #Set bifurcation parameters
        model.parameter_list=a_list

        #Compute trajectories using forward pass
        pred_x_ic1 = odeint(model, ic_list1, t,rtol=1e-5)
        pred_x_ic2 = odeint(model, ic_list2, t,rtol=1e-5)

        max_val_ic1,_ = torch.max(pred_x_ic1[-range_l:,:,0], dim=0)
        min_val_ic1,_ = torch.min(pred_x_ic1[-range_l:,:,0], dim=0)

        max_val_ic2,_ = torch.max(pred_x_ic2[-range_l:,:,0], dim=0)
        min_val_ic2,_ = torch.min(pred_x_ic2[-range_l:,:,0], dim=0)

        print(f"Saving to {filename}")
        torch.save(
            {
            "a_points": a_list,
            "max_val_ic1": max_val_ic1,
            "min_val_ic1": min_val_ic1,
            "max_val_ic2": max_val_ic2,
            "min_val_ic2": min_val_ic2,
            }
        , filename)
        print("File saved successfully.")

        if visual:
            #Visualise points:

            plt.figure(figsize=(7,6))
            
            plt.scatter(a_list,max_val_ic1,color="black",s=1)
            plt.scatter(a_list,min_val_ic1,color="black",s=1)
            plt.scatter(a_list,max_val_ic2,color="black",s=1)
            plt.scatter(a_list,min_val_ic2,color="black",s=1)
            
            plt.xlim((0.6,0.75))
            plt.ylim((-0.1,0.7))
            plt.xlabel(r"$\alpha$")
            plt.ylabel("x")
            plt.title("True system dynamics")
            plt.show()