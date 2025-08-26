import torch.nn as nn
from torchdiffeq import odeint
import torch
from math import comb
import numpy as np

class Neuralode(nn.Module):

    def __init__(self,
                 variables, 
                 drivers, 
                 parameter_list=[],
                 hidden_layers=1,
                 depth_of_layers=10,
                 ):
        super().__init__()

        #Define attributes of our NeuralODE
        self.number_of_features = variables+drivers
        self.depth              = depth_of_layers
        self.hiddenlayers       = hidden_layers
        self.drivers            = drivers
        self.variables          = variables
        self.parameter_list     = parameter_list

        #Build the network of hidden layers with given depth and tangent/sigmoid activation
        layers = []
        previous_size=self._compute_input_dim()
        for _ in range(self.hiddenlayers):
            layers.append(nn.Linear(previous_size,self.depth))
            layers.append(nn.Tanh())
            previous_size=self.depth
        layers.append(nn.Linear(previous_size,variables))    
        self.net = nn.Sequential(*layers)

        #Initialise weights
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.2)
                nn.init.constant_(m.bias, val=0)

    #Compute size of input layer
    def _compute_input_dim(self):
        """Calculate input dimension for the network."""
        return self.number_of_features

    def forward(self, t, x):
        parameters =self.parameter_list.float()
        #Merge state x and variables a
        input = torch.cat((x, parameters.unsqueeze(1)), dim=1)
        #Forward pass through network
        output = self.net(input)
        return output