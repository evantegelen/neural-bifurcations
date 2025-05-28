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
This code can be used to generate the points necessary to generate the bifurcation diagram for the true system.
The idea is to generate trajectorie data for different initial conditions and extrapolate the convergence behavior.
"""

visual=True

filename = "Figures/bifurcation_points/true.pth"

def population(q,t):
    #The variables
    x=q[0]
    y=q[1]
    a=q[2]
    #Define equations
    dxdt=3*x*(1-x)-x*y-a*(1-exp(-5*x))
    dydt=-y+3*x*y
    dadt=0
    return [dxdt,dydt,dadt]

#Time-span on which to calculate trajectories
t = torch.from_numpy(np.linspace(0,2000,2000)).float()
#Number of alpha samples to use
N_a=500

range_l=500 #Last ... timepoints to select min and max from.
a_list= np.linspace(0.6,0.75,N_a)

min_ic1,max_ic1,min_ic2,max_ic2=[],[],[],[]

for a0 in a_list:
    data_ic1 = sc_int.odeint(population,[1,0.01,a0],t)
    data_ic2 = sc_int.odeint(population,[0,0,a0],t)
    
    xmin_ic1=np.min(data_ic1[-range_l:,0])
    xmax_ic1=np.max(data_ic1[-range_l:,0])
    xmin_ic2=np.min(data_ic2[-range_l:,0])
    xmax_ic2=np.max(data_ic2[-range_l:,0])
    
    min_ic1.append(xmin_ic1)
    min_ic2.append(xmin_ic2)
    max_ic1.append(xmax_ic1)
    max_ic2.append(xmax_ic2)

print(f"Saving to {filename}")
torch.save(
    {
    "a_points": torch.from_numpy(a_list),
    "max_val_ic1": torch.tensor(max_ic1),
    "min_val_ic1": torch.tensor(min_ic1),
    "max_val_ic2": torch.tensor(max_ic2),
    "min_val_ic2": torch.tensor(min_ic2),
    }
, filename)
print("File saved successfully.")
      
if visual:
    #Visualise points:

    plt.figure(figsize=(7,6))
    
    plt.scatter(a_list,max_ic1,color="black",s=1)
    plt.scatter(a_list,min_ic1,color="black",s=1)
    plt.scatter(a_list,max_ic2,color="black",s=1)
    plt.scatter(a_list,min_ic2,color="black",s=1)
    
    plt.xlim((0.6,0.75))
    plt.ylim((-0.1,0.7))
    plt.xlabel(r"$\alpha$")
    plt.ylabel("x")
    plt.title("True system dynamics")
    plt.show()