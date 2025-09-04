import matplotlib.pyplot as plt
import scipy.integrate as sc_int
import numpy as np
import warnings
import torch
from math import exp
import random
import itertools
import os

warnings.simplefilter('always', UserWarning)

#We define our population model:

def population(q,t):
    #The variables
    x,y,a=q
    #Define equations
    dxdt=3*x*(1-x)-x*y-a*(1-exp(-5*x))
    dydt=-y+3*x*y
    dadt=0 
    return [dxdt,dydt,dadt]

def create_data(time,initial_conditions,noise_level=0):
    random.seed(124)
    data=[]
    for i in range(len(initial_conditions)):
        x0,y0,a0=initial_conditions[i,:]
        trajectory_data=sc_int.odeint(population,[x0,y0,a0],time)
        
        #Measurement noise is added to data
        if noise_level!=0:
            measurement_noise= np.random.normal(0,noise_level,[len(time),2])
            trajectory_data[:,:2]=trajectory_data[:,:2]*(measurement_noise+1)
        
        data.append(torch.from_numpy(trajectory_data[:,:]))
    data=torch.stack(tuple(data),dim=1) #Change to torch tensor
    return data

def export_data(time,data,filename):

    if os.path.exists(filename):
        overwrite = input(f"File {filename} already exists. Do you want to overwrite it? (y/n): ").strip().lower()
        if overwrite != 'y':
            print("Save operation canceled.")
            return

    print(f"Saving to {filename}")
    torch.save(
        {
        "data_x": data,
        "data_t": torch.from_numpy(time)
        }
    , filename)
    print("File saved successfully.")


if __name__ == "__main__":

    """
    For each of our datasets we use the following scheme to generate and save the data:

    #Define initial conditions
    x0_list=[...]
    y0_list=[...]
    a0_list=[...]

    initial_conditions = list(itertools.product(a0_list, y0_list, x0_list))         #We use each combination of initial conditions
    initial_conditions = np.array(initial_conditions)[:, [2, 1, 0]] #[x,y,a]

    #Create data from initial conditions
    data_set = create_data(t,initial_conditions)
    
    export_data(time,data_set,data_set name)

    """

    t = np.linspace(0,100,100)

    ###Primary dataset###
    data_name="Data/primary_dataset.pth"
    x0_list=[1]
    y0_list=[0.01,0.1]
    a0_list=[0.62,0.64,0.66,0.68,0.7]

    initial_conditions = list(itertools.product(a0_list, y0_list, x0_list))
    initial_conditions = np.array(initial_conditions)[:, [2, 1, 0]] #[x,y,a]

    data = create_data(t,initial_conditions)
    export_data(t,data,data_name)

    ###Tuning datasets###
    data_name="Data/tuning_training.pth"
    x0_list=[1]
    y0_list=[0.01,0.1]
    a0_list=[0.64,0.66,0.68,0.7]

    initial_conditions = list(itertools.product(a0_list, y0_list, x0_list))
    initial_conditions = np.array(initial_conditions)[:, [2, 1, 0]] #[x,y,a]

    data = create_data(t,initial_conditions)
    export_data(t,data,data_name)

    data_name="Data/tuning_validation.pth"
    x0_list=[1]
    y0_list=[0.01,0.1]
    a0_list=[0.62]

    initial_conditions = list(itertools.product(a0_list, y0_list, x0_list))
    initial_conditions = np.array(initial_conditions)[:, [2, 1, 0]] #[x,y,a]

    data = create_data(t,initial_conditions)
    export_data(t,data,data_name)

    #Experiment 1: Different regimes
    data_name="Data/exp1_onlyB.pth"
    x0_list=[1]
    y0_list=[0.01,0.1]
    a0_list=[0.68,0.685,0.69,0.695,0.7]

    initial_conditions = list(itertools.product(a0_list, y0_list, x0_list))
    initial_conditions = np.array(initial_conditions)[:, [2, 1, 0]] #[x,y,a]

    data = create_data(t,initial_conditions)
    export_data(t,data,data_name)

    data_name="Data/exp1_onlyA.pth"
    x0_list=[1]
    y0_list=[0.01,0.1]
    a0_list=[0.61,0.62,0.63,0.64,0.65]

    initial_conditions = list(itertools.product(a0_list, y0_list, x0_list))
    initial_conditions = np.array(initial_conditions)[:, [2, 1, 0]] #[x,y,a]

    data = create_data(t,initial_conditions)
    export_data(t,data,data_name)

    ###Experiment 2: Limited data###
    data_name="Data/exp2_limit_alpha.pth"
    x0_list=[1]
    y0_list=[0.01,0.1]
    a0_list=[0.62,0.66,0.7]

    initial_conditions = list(itertools.product(a0_list, y0_list, x0_list))
    initial_conditions = np.array(initial_conditions)[:, [2, 1, 0]] #[x,y,a]

    data = create_data(t,initial_conditions)
    export_data(t,data,data_name)    

    data_name="Data/exp2_limit_ic.pth"
    x0_list=[1]
    y0_list=[0.1]
    a0_list=[0.62,0.64,0.66,0.68,0.7]

    initial_conditions = list(itertools.product(a0_list, y0_list, x0_list))
    initial_conditions = np.array(initial_conditions)[:, [2, 1, 0]] #[x,y,a]

    data = create_data(t,initial_conditions)
    export_data(t,data,data_name)

    data_name="Data/exp2_increase_alpha.pth"
    x0_list=[1]
    y0_list=[0.01,0.1]
    a0_list=[0.61,0.62,0.63,0.64,0.65,0.66,0.67,0.68,0.69,0.7]

    initial_conditions = list(itertools.product(a0_list, y0_list, x0_list))
    initial_conditions = np.array(initial_conditions)[:, [2, 1, 0]] #[x,y,a]

    data = create_data(t,initial_conditions)
    export_data(t,data,data_name)    

    ###Experiment 3: Noise levels###
    noiselevel=0.2
    data_name=f"Data/exp3_noise{noiselevel}.pth"

    x0_list=[1]
    y0_list=[0.01,0.1]
    a0_list=[0.62,0.64,0.66,0.68,0.7]

    initial_conditions = list(itertools.product(a0_list, y0_list, x0_list))
    initial_conditions = np.array(initial_conditions)[:, [2, 1, 0]] #[x,y,a]

    data = create_data(t,initial_conditions,noise_level=noiselevel)
    export_data(t,data,data_name)

    noiselevel=0.05
    data_name=f"Data/exp3_noise{noiselevel}.pth"

    x0_list=[1]
    y0_list=[0.01,0.1]
    a0_list=[0.62,0.64,0.66,0.68,0.7]

    initial_conditions = list(itertools.product(a0_list, y0_list, x0_list))
    initial_conditions = np.array(initial_conditions)[:, [2, 1, 0]] #[x,y,a]

    data = create_data(t,initial_conditions,noise_level=noiselevel)
    export_data(t,data,data_name)
