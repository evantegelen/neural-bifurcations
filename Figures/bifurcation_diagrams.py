import matplotlib.pyplot as plt
import scipy.integrate as sc_int
import numpy as np
import warnings
import torch
from math import exp
import random
warnings.simplefilter('always', UserWarning)
from torchdiffeq import odeint
import sys
import argparse
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

#Load the true datapoints
true_points_name = "Figures/bifurcation_points/true.pth"
data_true = torch.load(true_points_name)

max_x_true=data_true["max_val_ic1"]
min_x_true=data_true["min_val_ic1"]
max_x_true_2=data_true["max_val_ic2"]
min_x_true_2=data_true["min_val_ic2"]
a_points = data_true["a_points"]

#Which experiment do you want to consider:
experiment ="ex2_increase"

#First we print mean absolute error and select seed with smallest one:
current_best_error = 10
current_best_model = 0
for i in range(5):
    predicted_points_name   = f"Figures/bifurcation_points/{experiment}_{i+1}.pth"
    #Load the data
    data_predicted = torch.load(predicted_points_name)

    max_x_predicted=data_predicted["max_val_ic1"]
    min_x_predicted=data_predicted["min_val_ic1"]
    a_points_predicted = data_predicted["a_points"]

    error_max =torch.mean(abs(max_x_predicted-max_x_true))
    error_min =torch.mean(abs(min_x_predicted-min_x_true))
    total_error=(error_max+error_min)*0.5

    if total_error<current_best_error:
        current_best_model=i+1
        current_best_error=total_error
        
    print(f"MAE for run {i+1} = {total_error}")
print(f"So lowest MAE is for run: {current_best_model}")

#Then we can plot the figure where seed with lowest MAE is main focus
plt.figure(figsize=(7,5))
color="#0F0F0F"
s=2
#First plot true points
plt.scatter(a_points,min_x_true,color=color,s=s,zorder=5)
plt.scatter(a_points,max_x_true,color=color,s=s,zorder=5)
plt.scatter(a_points,min_x_true_2,color=color,s=s,zorder=5)
plt.scatter(a_points,max_x_true_2,color=color,s=s,zorder=5)
for i in range(5):
    predicted_points_name   = f"Figures/bifurcation_points/{experiment}_{i+1}.pth"
    #Load the predicted data
    data_predicted = torch.load(predicted_points_name)

    max_x_predicted=data_predicted["max_val_ic1"]
    min_x_predicted=data_predicted["min_val_ic1"]
    max_x_predicted_2=data_predicted["max_val_ic2"]
    min_x_predicted_2=data_predicted["min_val_ic2"]
    a_points_predicted = data_predicted["a_points"]

    if i+1==current_best_model: #set color and size for best model
        color = "#374B94"
        zorder = 6
        s=2
    else:
        color = "#ACB1B0" #...and for the other models
        zorder = 4
        s =0.5

    plt.scatter(a_points_predicted,min_x_predicted,color=color,s=s,zorder=zorder)
    plt.scatter(a_points_predicted,max_x_predicted,color=color,s=s,zorder=zorder)
    plt.scatter(a_points_predicted,min_x_predicted_2,color=color,s=s,zorder=zorder)
    plt.scatter(a_points_predicted,max_x_predicted_2,color=color,s=s,zorder=zorder)

plt.xlim((0.6,0.75))
plt.ylim((-0.1,0.7))
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.savefig(f"Figures/bifurcation_diagrams/{experiment}.png", transparent=True)
plt.show()
