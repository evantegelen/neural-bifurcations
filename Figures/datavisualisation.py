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

#Load the data
#dataname=f"Data/primary_dataset.pth"
dataname=f"Data/exp3_noise0.05.pth"

data = torch.load(dataname)
data_x = data["data_x"][:,:,:-1].float()
data_a = data["data_x"][0,:,-1].float()
data_t = data["data_t"].float()

plt.figure(figsize=(6,2.4))
plt.scatter(data_t,data_x[:,1,0],color="#DC267F",s=7)
plt.scatter(data_t,data_x[:,1,1],color="#503F9E",s=7)
plt.xlabel("t")
plt.tight_layout()
plt.ylim(0,1)
plt.savefig(f"Figures/data/timeseries_lownoise.png", transparent=True)
plt.show()